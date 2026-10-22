"""Deterministic synthetic regression tests; not a biological accuracy benchmark."""
import unittest
import numpy as np
import importlib.util
from pathlib import Path
import sys, tempfile, json, subprocess
import tifffile as tf
path = Path(__file__).resolve().parents[1] / "track.py"
spec = importlib.util.spec_from_file_location("released_track", path)
track = importlib.util.module_from_spec(spec); sys.modules[spec.name] = track; spec.loader.exec_module(track)
repair_links = track._evidence_repair_links
restore_gaps = track._evidence_restore_gaps
metrics = track._evidence_metrics

class StationaryMotion:
    spacing=np.array([.1,.1,.1])
    def transport(self,points,t,q):return points.copy(),np.zeros(len(points))

class RepairTests(unittest.TestCase):
    def setUp(self):self.motion=StationaryMotion();self.sp=self.motion.spacing

    def test_raw_supported_identity_swap(self):
        v=np.zeros((6,12,16,52),np.uint16)
        for t in range(6):
            v[t,3:8,4:10,5:11]=1 if t<3 else 2
            v[t,3:8,4:10,35:41]=2 if t<3 else 1
        out,events,uncertain,paths=repair_links(v,self.sp,self.motion)
        self.assertTrue(np.array_equal(v>0,out>0))
        self.assertEqual(len(set(out[:,5,6,8])),1)
        self.assertEqual(len(set(out[:,5,6,38])),1)
        self.assertNotEqual(out[0,5,6,8],out[0,5,6,38])
        self.assertTrue(any(e['type']=='raw_bidirectional_alias_or_swap_repaired' for e in events))

    def test_long_gap_not_arbitrarily_split(self):
        v=np.zeros((6,12,16,20),np.uint16);v[0,3:8,4:10,5:11]=7;v[5,3:8,4:10,5:11]=7
        out,events,uncertain,paths=repair_links(v,self.sp,self.motion)
        self.assertEqual(out[0,5,6,8],out[5,5,6,8])
        self.assertTrue(any(e['type']=='long_unobserved_interval' for e in uncertain))

    def test_bracketed_merge_restores_existing_foreground(self):
        v=np.zeros((6,12,16,24),np.uint16)
        v[:,3:8,4:10,5:11]=1;v[:,3:8,4:10,11:20]=2
        v[3][v[3]==1]=2
        out,events,rejected=restore_gaps(v,self.sp,self.motion)
        self.assertTrue(np.array_equal(v>0,out>0))
        self.assertTrue(np.any(out[3]==1));self.assertTrue(events)
        self.assertEqual(metrics(out,self.sp,self.motion)['interior_gap_frames'],0)

    def test_no_unobserved_foreground_is_created(self):
        v=np.zeros((6,12,16,24),np.uint16);v[:,3:8,4:10,5:11]=1;v[3]=0
        out,events,rejected=restore_gaps(v,self.sp,self.motion)
        self.assertTrue(np.array_equal(v,out));self.assertFalse(events)

    def test_empty_movie_is_explicitly_empty(self):
        v=np.zeros((6,12,16,24),np.uint16)
        out,events,uncertain,paths=repair_links(v,self.sp,self.motion)
        self.assertTrue(np.array_equal(v,out));self.assertFalse(paths)
        self.assertEqual(metrics(out,self.sp,self.motion)['median_track_frames'],0.)


    def test_shaped_and_ome_calibration_and_geometry_roundtrip(self):
        for ome in (False, True):
            with self.subTest(ome=ome), tempfile.TemporaryDirectory() as tmp:
                source=Path(tmp)/"source.tif"; output=Path(tmp)/"out.tif"
                v=np.zeros((3,4,8,9),np.uint16); v[:,1:3,2:6,2:7]=1
                provenance={"origin_zyx_um":[1.,2.,3.],"time_interval_seconds":30.}
                meta=dict(axes="TZYX",**provenance)
                if ome:
                    meta=dict(axes="TZYX",PhysicalSizeZ=.3,PhysicalSizeY=.1,PhysicalSizeX=.1,
                        PhysicalSizeZUnit="µm",PhysicalSizeYUnit="µm",PhysicalSizeXUnit="µm",
                        TimeIncrement=30.,TimeIncrementUnit="s",Description=json.dumps(provenance))
                else: meta.update(unit="micron",spacing=.3)
                tf.imwrite(source,v,ome=ome,metadata=meta,resolution=(10,10),resolutionunit="NONE",photometric="minisblack")
                self.assertTrue(np.allclose(track._read_metadata(source)[3],[.3,.1,.1]))
                track._write_evidence_tiff(source,output,v,{"version":track.EVIDENCE_VERSION})
                actual=track._evidence_read_movie(output)
                self.assertTrue(np.array_equal(v,actual))
                saved=track._read_metadata(output)[0]
                self.assertEqual(saved["origin_zyx_um"],[1.,2.,3.])
                self.assertEqual(saved["tracking_evidence_repair"]["version"],track.EVIDENCE_VERSION)
                if ome: self.assertEqual(saved["finterval"],30.)

    def test_no_inplace_source_overwrite(self):
        with self.assertRaisesRegex(ValueError,"read-only"):
            track.repair_existing("same.tif","raw.tif","post.tif","same.tif","cache","audit.json")

    def test_invalid_axes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"q.tif"
            tf.imwrite(path,np.zeros((3,8,8),np.uint16),metadata={"axes":"QYX"},photometric="minisblack")
            with self.assertRaises(ValueError): track._evidence_read_movie(path)

    def test_real_flow_cache_and_array_entry(self):
        v=np.zeros((4,8,16,24),np.uint16);v[:,2:6,4:12,4:12]=1
        raw=(v>0).astype(np.uint16)*1000
        with tempfile.TemporaryDirectory() as tmp:
            out,report=track.evidence_repair(v,raw,v,np.array([.3,.1,.1]),tmp,'synthetic-raw')
            self.assertTrue(np.array_equal(v>0,out>0))
            self.assertTrue((Path(tmp)/'f00.npy').exists())
            self.assertEqual(report['foreground_added'],0)
            # Cache rejects even shape-compatible, different raw data.
            with self.assertRaisesRegex(AssertionError,'stale'):
                track.evidence_repair(v,raw,v,np.array([.3,.1,.1]),tmp,'different-raw')

    def test_cli_repair_and_default_raw_aware_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ('raw','post','old'): (root/folder).mkdir()
            v=np.zeros((5,12,28,48),np.uint16)
            zz,yy,xx=np.indices(v.shape[1:])
            for t in range(len(v)):
                v[t][((zz-5)/3)**2+((yy-13)/5)**2+((xx-12)/5)**2<1]=1
                v[t][((zz-5)/3)**2+((yy-13)/5)**2+((xx-34)/5)**2<1]=2
            for folder,name,data in [('post','sample.tif',v),('old','sample.tif',v),('raw','sample-DAPI.tif',(v>0).astype(np.uint16)*2000)]:
                tf.imwrite(root/folder/name,data,photometric='minisblack',
                    metadata=dict(axes='TZYX',unit='micron',spacing=.2),resolution=(10,10))
            for extra,out in [(['--repair-existing','--initial-tracks',str(root/'old')],'repair'),([], 'full')]:
                proc=subprocess.run([sys.executable,str(path),'--input',str(root/'post'),'--raw',str(root/'raw'),
                    '--output',str(root/out),*extra],capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                audit=json.loads((root/out/'reports/sample.json').read_text())
                if out=='repair': self.assertEqual(audit['version'],track.EVIDENCE_VERSION)
                else:
                    self.assertEqual(audit['version'],track.VERSION)
                    self.assertEqual(audit['evidence_repair']['version'],track.EVIDENCE_VERSION)

if __name__=='__main__':unittest.main()
