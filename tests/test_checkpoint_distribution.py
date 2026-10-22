"""Repository distribution policy checks; no downloaded weights/GPU required."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = ('.pth', '.pt', '.ckpt', '.h5', '.hdf5', '.safetensors', '.onnx')


class CheckpointDistributionTests(unittest.TestCase):
    def test_no_checkpoint_binaries_tracked(self):
        files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        embedded = [p for p in files if p.lower().endswith(WEIGHT_SUFFIXES)
                    or Path(p).name.lower() == 'cpsam']
        self.assertEqual(embedded, [])

    def test_downloaded_binaries_are_ignored_outside_checkpoint_directory(self):
        paths = ['test-downloads/model'+suffix for suffix in WEIGHT_SUFFIXES]
        paths.append('test-downloads/cpsam')
        result = subprocess.run(['git', 'check-ignore', '--stdin', '-z'], cwd=ROOT,
            input=('\0'.join(paths)+'\0').encode(), capture_output=True, check=True)
        self.assertEqual(set(result.stdout.decode().rstrip('\0').split('\0')), set(paths))

    def test_external_model_guides_have_no_project_drive_downloads(self):
        guide = (ROOT/'docs/downloads.md').read_text()
        external = guide.split('## DINOv3 — ChromoPix backbone', 1)[1]
        external += (ROOT/'Checkpoint/Outsource/README.md').read_text()
        self.assertNotRegex(external, r'https?://(?:drive|docs)\.google\.com/')
        for source in ('github.com/MouseLand/cellpose', 'github.com/CSBDeep/CSBDeep',
                       'github.com/juglab/EmbedSeg', 'github.com/voxelmorph/voxelmorph',
                       'github.com/facebookresearch/dinov3'):
            self.assertIn(source, external)

    def test_cpsam_visualization_directories_are_not_ignored(self):
        # The extensionless weight is called cpsam, but demo folders may
        # legitimately use that model name too. Only exclude the file.
        result = subprocess.run(['git', 'check-ignore', 'test-downloads/cpsam/preview.png'],
                                cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 1, result.stdout.decode())

    def test_flat_internal_download_names_and_runtime_mapping(self):
        guide = (ROOT/'docs/downloads.md').read_text()
        for filename in ('chromopix.pth', 'chromopose.pth', 'chromolift.pth'):
            self.assertRegex(guide, re.escape(filename)+r'\]\(https://drive\.google\.com/file/d/')
        for target in ('Checkpoint/ChromoPix/chromopix.pth',
                       'Checkpoint/ChromoPhy/ChromoPose/last.pth',
                       'Checkpoint/ChromoPhy/ChromoLift/lifter_final.pth'):
            self.assertIn(target, guide)
        self.assertIn('1B4LLQiesGVCVG3aq-7846YpRV52jHJg3', guide)


if __name__ == '__main__':
    unittest.main()
