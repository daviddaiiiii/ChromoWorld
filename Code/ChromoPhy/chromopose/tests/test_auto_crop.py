"""CPU-only synthetic regressions: python -m unittest discover -s .../tests."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile as tf

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import auto_crop as ac

PIPELINE = HERE.parents[1]/"Analyze/fig5/get-4dmask_TIFF2TIFF.py"
spec = importlib.util.spec_from_file_location("crop_pipeline", PIPELINE)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class AutoCropTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="auto-crop-test-")
        self.root = Path(self.temp.name)
        self.config = ac.CropConfig(margin_xy_um=.2, margin_z_um=.4, square_xy=False)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data, axes="TZYX", origin=None, ome=False):
        p = self.root/name
        if ome:
            meta = dict(axes=axes, PhysicalSizeZ=.4, PhysicalSizeY=.1, PhysicalSizeX=.1,
                        PhysicalSizeZUnit="µm", PhysicalSizeYUnit="µm", PhysicalSizeXUnit="µm",
                        TimeIncrement=30., TimeIncrementUnit="s")
        else:
            meta = dict(axes=axes, unit="micron", spacing=.4, finterval=30.)
            if origin is not None:
                meta["origin_zyx_um"] = origin
        tf.imwrite(p, data, photometric="minisblack", compression="zlib", ome=ome,
                   metadata=meta, resolution=(10, 10), resolutionunit="NONE")
        return p

    def data(self):
        a = np.full((3, 28, 64, 64), 40, np.uint16)
        for t in range(3):
            a[t, 9:15, 20+t*3:27+t*3, 20:28] = 1000
        # Small, well separated second object: never discarded merely for
        # having a small fraction of the largest component's volume.
        a[2, 19:21, 44:46, 43:45] = 350
        return a

    def assert_exact(self, source, output, report):
        start, stop = report["crop_start_zyx"], report["crop_stop_zyx"]
        a = ac.Movie(source)
        b = ac.Movie(output)
        try:
            self.assertEqual(a.axes, b.axes)
            self.assertEqual(a.tzyx[0], b.tzyx[0])
            self.assertEqual(a.dtype, b.dtype)
            for t in range(a.tzyx[0]):
                np.testing.assert_array_equal(a.read(t)[ac._slices(a,start,stop)], b.read(t))
            np.testing.assert_allclose(a.spacing,b.spacing)
            np.testing.assert_allclose(b.origin,a.origin+np.array(start)*a.spacing)
        finally:
            a.close(); b.close()

    def test_all_time_union_native_pixels_and_small_daughter(self):
        data=self.data();p=self.write("raw.tif",data)
        original=p.read_bytes()
        report=ac.crop_file(p,self.root/"out",self.config)
        self.assertEqual(p.read_bytes(),original)
        self.assert_exact(p,self.root/"out/raw.tif",report)
        lo,hi=report["crop_start_zyx"],report["crop_stop_zyx"]
        self.assertLessEqual(lo[0],9);self.assertGreaterEqual(hi[0],21)
        self.assertGreaterEqual(hi[1],46);self.assertGreaterEqual(hi[2],45)
        self.assertLess(hi[0]-lo[0],data.shape[1])
        for frame in report["frames"]:
            self.assertGreaterEqual(frame.get("retained_signal",1),.998-1e-7)

    def test_blank_movie_is_identity_and_time_is_retained(self):
        p=self.write("blank.tif",np.zeros((3,8,16,16),np.uint16))
        d=ac.crop_file(p,self.root/"out",self.config)
        self.assertEqual(d["crop_start_zyx"],[0,0,0]);self.assertEqual(d["crop_stop_zyx"],[8,16,16])
        self.assert_exact(p,self.root/"out/blank.tif",d)

    def test_noise_only_frame_forces_conservative_full_box(self):
        rng=np.random.default_rng(17)
        image=rng.normal(100,3,(18,40,40)).astype(np.float32)
        bounds,info=ac.signal_bounds(image,self.config)
        self.assertEqual(info["status"],"unresolved_keep_full_frame")
        np.testing.assert_array_equal(bounds[0],[0,0,0]);np.testing.assert_array_equal(bounds[1],image.shape)

    def test_companion_labels_apply_same_crop_and_keep_ids(self):
        data=self.data();raw=self.write("raw.tif",data,origin=[1.,2.,3.])
        labels=np.where(data>100,70001,0).astype(np.uint32)
        mask=self.write("mask.tif",labels,origin=[1.,2.,3.])
        d=ac.crop_file(raw,self.root/"out",self.config,apply_to=[mask])
        self.assert_exact(mask,self.root/"out/mask.tif",d)
        self.assertEqual(int(tf.imread(self.root/"out/mask.tif").max()),70001)

    def test_companion_mismatched_grid_and_origin_rejected(self):
        p=self.write("raw.tif",self.data(),origin=[1.,0.,0.])
        q=self.write("wrong.tif",self.data(),origin=[2.,0.,0.])
        with self.assertRaisesRegex(ValueError,"origin"):
            ac.crop_file(p,self.root/"out",self.config,apply_to=[q])
        q=self.write("wrong_shape.tif",self.data()[:,:,:-1],origin=[1.,0.,0.])
        with self.assertRaisesRegex(ValueError,"grid"):
            ac.crop_file(p,self.root/"out",self.config,apply_to=[q])
        self.assertFalse((self.root/"out").exists())

    def test_ome_metadata_read_and_downstream_calibration(self):
        p=self.write("ome.ome.tif",self.data(),ome=True)
        d=ac.crop_file(p,self.root/"out",self.config)
        self.assert_exact(p,self.root/"out/ome.ome.tif",d)
        with tf.TiffFile(self.root/"out/ome.ome.tif") as f:
            meta=json.loads(f.pages[0].description)
            self.assertEqual(meta["finterval"],30.)
            self.assertEqual(meta["origin_reference"],"local_source_voxel_grid")
        # Auto-crop normalizes OME calibration to the established shaped-TIFF
        # interface, so the maintained preprocess and affine readers accept it.
        import preprocess
        import affine
        with tf.TiffFile(self.root/"out/ome.ome.tif") as f:
            np.testing.assert_allclose(preprocess.metadata(f)[0],[.4,.1,.1])
            np.testing.assert_allclose(affine.get_metadata(f)[0],[.4,.1,.1])

    def test_multichannel_requires_explicit_dapi_and_retains_both(self):
        a=self.data();data=np.stack([a,np.arange(a.size,dtype=np.uint16).reshape(a.shape)],axis=1)
        p=self.write("channels.tif",data,axes="TCZYX")
        with self.assertRaisesRegex(ValueError,"explicit --channel"):
            ac.crop_file(p,self.root/"out",self.config)
        d=ac.crop_file(p,self.root/"out",self.config,channel=0)
        self.assert_exact(p,self.root/"out/channels.tif",d)
        self.assertEqual(tf.imread(self.root/"out/channels.tif").shape[1],2)

    def test_decimal_ome_resolution_roundtrip(self):
        p=self.root/'decimal.ome.tif'
        tf.imwrite(p,self.data(),ome=True,photometric='minisblack',metadata=dict(
            axes='TZYX',PhysicalSizeZ=.326,PhysicalSizeY=.06,PhysicalSizeX=.06))
        d=ac.crop_file(p,self.root/'out',self.config)
        self.assert_exact(p,self.root/'out'/p.name,d)

    def test_multi_series_cannot_silently_drop_other_images(self):
        p=self.root/'multiple.tif'
        with tf.TiffWriter(p) as writer:
            for i in range(2):
                writer.write(self.data()+i,photometric='minisblack',metadata=dict(
                    axes='TZYX',unit='micron',spacing=.4),resolution=(10,10))
        with self.assertRaisesRegex(ValueError,'Multi-series'):
            ac.crop_file(p,self.root/'out',self.config)
        self.assertFalse((self.root/'out').exists())

    def test_zcyx_and_zyx_axes(self):
        for axes,data in [("TZCYX",np.stack([self.data(),self.data()],axis=2)),
                          ("ZYX",self.data()[0])]:
            p=self.write(axes+".tif",data,axes=axes)
            folder=self.root/("out_"+axes)
            d=ac.crop_file(p,folder,self.config,channel=0)
            self.assert_exact(p,folder/p.name,d)

    def test_z_only_and_edge_margin_clamping(self):
        data=self.data();data[0,0:3,0:4,0:4]=700
        p=self.write("edge.tif",data)
        cfg=ac.CropConfig(margin_xy_um=1,margin_z_um=.5,axes_to_crop="z")
        d=ac.crop_file(p,self.root/"out",cfg)
        self.assertEqual(d["crop_start_zyx"],[0,0,0])
        self.assertEqual(d["crop_stop_zyx"][1:],[64,64])
        self.assert_exact(p,self.root/"out/edge.tif",d)

    def test_missing_spacing_and_ambiguous_axes_need_override(self):
        p=self.root/"generic.tif"
        tf.imwrite(p,self.data(),photometric="minisblack",metadata={"axes":"QQYX"})
        with self.assertRaisesRegex(ValueError,"axes"):
            ac.crop_file(p,self.root/"out",self.config)
        with self.assertRaisesRegex(ValueError,"spacing"):
            ac.crop_file(p,self.root/"out",self.config,axes="TZYX")
        d=ac.crop_file(p,self.root/"out",self.config,axes="TZYX",spacing=[.4,.1,.1])
        self.assertEqual(d["source_shape_tzyx"],list(self.data().shape))

    def test_aligned_companion_missing_origin_inherits_explicit_dapi_grid(self):
        p=self.write("raw.tif",self.data(),origin=[-2.,3.,4.])
        q=self.write("mask.tif",np.where(self.data()>100,7,0).astype(np.uint16))
        d=ac.crop_file(p,self.root/"out",self.config,apply_to=[q])
        a=ac.Movie(self.root/"out/raw.tif");b=ac.Movie(self.root/"out/mask.tif")
        try:np.testing.assert_allclose(a.origin,b.origin)
        finally:a.close();b.close()
        self.assertEqual(d['files'][1]['origin_inherited_from'],str(p.resolve()))

    def test_physical_square_and_output_publication_failure_cleanup(self):
        p=self.write("raw.tif",self.data())
        d=ac.crop_file(p,self.root/"out",ac.CropConfig(margin_xy_um=0,margin_z_um=0))
        self.assertEqual(d['output_shape_tzyx'][2],d['output_shape_tzyx'][3])
        real_link=ac.os.link;called=[]
        def fail_second(src,dst):
            called.append(dst)
            if len(called)==2:raise OSError('synthetic publication failure')
            real_link(src,dst)
        with patch.object(ac.os,'link',side_effect=fail_second):
            with self.assertRaisesRegex(OSError,'publication failure'):
                ac.crop_file(p,self.root/'failed',self.config)
        self.assertEqual(list((self.root/'failed').iterdir()),[])
        self.assertTrue(p.exists())

    def test_never_overwrite_source_or_previous_output(self):
        p=self.write("raw.tif",self.data())
        with self.assertRaisesRegex(ValueError,"overwrite"):
            ac.crop_file(p,self.root,self.config)
        ac.crop_file(p,self.root/"out",self.config)
        with self.assertRaises(FileExistsError):
            ac.crop_file(p,self.root/"out",self.config)

    def test_invalid_parameters_and_nonfinite_pixels(self):
        for c in [ac.CropConfig(margin_z_um=-1),ac.CropConfig(retained_signal=.2),ac.CropConfig(min_seed_voxels=0)]:
            with self.assertRaises(ValueError):c.validate()
        image=np.ones((4,5,5),np.float32);image[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            ac.signal_bounds(image,self.config)

    def test_pipeline_opt_in_routes_to_cropped_grids(self):
        source=self.write("sample-DAPI.tif",self.data())
        source_info=pipeline.tiff_info(source)
        args=pipeline.argparse.Namespace(output_dir=self.root/"pipeline",profile="legacy-v2",checkpoint=self.root/"absent.pth",
            gpu_ids="0",anchors=None,auto_crop=True,crop_margin_xy_um=.2,crop_margin_z_um=.4,
            affine_crop_margin_xy_um=.2,stop_after="affine_crop")
        calls=[]
        def fake_stage(sample_dir,stage,filename,producer,**kwargs):
            dest=sample_dir/stage;dest.mkdir(exist_ok=True)
            # Exercise command routing but do not need registration/model work.
            if stage in ("crop","preprocess","affine","affine_crop"):
                producer(dest)
            shape=source_info["shape_tzyx"].copy()
            if stage in ("crop","preprocess"):shape[1:]=[20,50,50]
            if stage in ("affine","affine_crop"):shape[1:]=[22,52,52]
            return dest/filename,dict(source_info,shape_tzyx=shape)
        with patch.object(pipeline,"finished_stage",side_effect=fake_stage),patch.object(pipeline,"run_command",side_effect=lambda *x:calls.append(x)):
            pipeline.process_sample(source,args,{})
        self.assertEqual([c[0].name for c in calls],["auto_crop.py","preprocess.py","affine.py","auto_crop.py"])
        pre=calls[1];self.assertEqual(Path(pre[pre.index("--input")+1]).parent.name,"crop")
        affine=calls[2];self.assertEqual(Path(affine[affine.index("--raw")+1]).parent.name,"crop")
        self.assertEqual(Path(affine[affine.index("--preprocessed")+1]).parent.name,"preprocess")
        self.assertIn("registered",calls[3])

    def test_full_pipeline_routes_post_and_inference_with_and_without_crop(self):
        source=self.write('sample-DAPI.tif',self.data())
        info=pipeline.tiff_info(source)
        checkpoint=self.root/'test.pth';checkpoint.write_bytes(b'test fixture, not a model')
        for enabled in (False,True):
            with self.subTest(auto_crop=enabled):
                args=pipeline.argparse.Namespace(output_dir=self.root/str(enabled),profile='legacy-v2',
                    checkpoint=checkpoint,gpu_ids='0',anchors=None,auto_crop=enabled,
                    crop_margin_xy_um=.2,crop_margin_z_um=.4,affine_crop_margin_xy_um=.2,stop_after='track')
                calls=[];stages=[]
                def fake_stage(sample_dir,stage,filename,producer,**kwargs):
                    stages.append(stage)
                    folder=sample_dir/stage;folder.mkdir(exist_ok=True)
                    if stage!='raw':producer(folder)
                    return folder/filename,info
                with patch.object(pipeline,'finished_stage',side_effect=fake_stage),patch.object(
                        pipeline,'run_command',side_effect=lambda *x:calls.append(x)):
                    pipeline.process_sample(source,args,{})
                expected='affine_crop' if enabled else 'affine'
                infer=next(c for c in calls if c[0].name=='infer.py')
                post=next(c for c in calls if c[0].name=='postprocess.py')
                self.assertEqual(Path(infer[infer.index('--in_path')+1]).parent.name,expected)
                self.assertEqual(Path(post[post.index('--raw')+1]).name,expected)
                self.assertEqual(stages,list(pipeline.STAGES) if enabled else
                                 ['raw','preprocess','affine','segment','post','track'])
                self.assertEqual(calls[-1][0].name,'track.py')

    def test_cli_end_to_end(self):
        p=self.write("arbitrary_basename.tif",self.data())
        run=subprocess.run([sys.executable,str(HERE/"auto_crop.py"),"--input",str(p),"--output-dir",str(self.root/"out"),
                            "--margin-xy-um","0.2","--margin-z-um","0.4"],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        report=json.loads((self.root/"out/arbitrary_basename_crop.json").read_text())
        self.assertTrue(report["timepoints_preserved"])
        self.assert_exact(p,self.root/"out/arbitrary_basename.tif",report)


if __name__=="__main__":
    unittest.main()
