import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

# Exercise the extraction script without importing the separate plotting tools.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downloadAvi"))
import extract_avi_metadata as extractor


class DicomExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_dicom(self, pixels, name="input", photo="MONOCHROME2", fps_tag=None):
        path = self.root / f"{name}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PhotometricInterpretation = photo
        ds.SamplesPerPixel = 3 if photo == "RGB" else 1
        if ds.SamplesPerPixel == 3:
            ds.PlanarConfiguration = 0
        spatial_shape = pixels.shape[:-1] if photo == "RGB" else pixels.shape
        ds.Rows, ds.Columns = spatial_shape[-2:]
        if len(spatial_shape) == 3:
            ds.NumberOfFrames = spatial_shape[0]
        ds.BitsAllocated = pixels.dtype.itemsize * 8
        ds.BitsStored = ds.BitsAllocated
        ds.HighBit = ds.BitsStored - 1
        ds.PixelRepresentation = 0
        if fps_tag is not None:
            ds.add_new(fps_tag, "IS", 5)
        ds.PixelData = pixels.tobytes()
        ds.save_as(str(path), write_like_original=False)
        return path

    def decode_video(self, path):
        cap = cv2.VideoCapture(str(path))
        self.assertTrue(cap.isOpened(), "Exported AVI must be readable")
        self.addCleanup(cap.release)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return cap, np.asarray(frames)

    def test_grayscale_screenshot_without_fps_is_png_with_real_metadata_path(self):
        pixels = np.arange(240, dtype=np.uint8).reshape(12, 20)
        dicom = self.write_dicom(pixels)
        metadata = json.loads(extractor.process_row(
            pd.Series({"path": str(dicom)}), str(self.root), None, "path"
        ))
        png = self.root / "input.png"
        self.assertEqual(metadata["video_path"], str(png))
        np.testing.assert_array_equal(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE), pixels)
        self.assertFalse((self.root / "input.avi").exists())

    def test_rgb_screenshot_keeps_colors(self):
        pixels = np.zeros((12, 20, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        dicom = self.write_dicom(pixels, photo="RGB")
        path, metadata = extractor.extract_h264_video_from_dicom(str(dicom), str(self.root / "rgb.avi"))
        self.assertEqual(metadata["video_path"], path)
        self.assertEqual(Path(path).suffix, ".png")
        np.testing.assert_array_equal(cv2.imread(path), pixels[..., ::-1])

    def test_nonsquare_angiography_video_round_trip_is_lossless(self):
        pixels = np.arange(5 * 12 * 20, dtype=np.uint8).reshape(5, 12, 20)
        dicom = self.write_dicom(pixels, fps_tag=(0x0008, 0x2144))
        path, metadata = extractor.extract_h264_video_from_dicom(str(dicom), str(self.root / "cine.avi"))
        self.assertNotIn("error", metadata)
        cap, decoded = self.decode_video(path)
        self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 5)
        np.testing.assert_array_equal(decoded, np.repeat(pixels[..., None], 3, axis=-1))

    def test_tte_fallback_tag_and_color_video_survive_merge(self):
        pixels = np.zeros((5, 12, 20, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(5, dtype=np.uint8)[:, None, None] * 40
        pixels[..., 1] = 100
        dicom = self.write_dicom(pixels, photo="RGB", fps_tag=(0x0018, 0x0040))
        path, metadata = extractor.extract_h264_video_from_dicom(
            str(dicom), str(self.root / "tte.avi"), data_type="TTE"
        )
        self.assertNotIn("error", metadata)
        cap, decoded = self.decode_video(path)
        self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 5)
        np.testing.assert_array_equal(decoded, pixels[..., ::-1])

    def test_missing_cine_rate_still_rejects_angiography_video(self):
        dicom = self.write_dicom(np.zeros((5, 12, 20), dtype=np.uint8))
        path, metadata = extractor.extract_h264_video_from_dicom(str(dicom), str(self.root / "cine.avi"))
        self.assertIsNone(path)
        self.assertEqual(metadata["error"], "No frame rate in DICOM")

    def test_closed_writer_is_error_without_success_path(self):
        dicom = self.write_dicom(np.zeros((5, 12, 20), dtype=np.uint8), fps_tag=(0x0008, 0x2144))
        writer = Mock()
        writer.isOpened.return_value = False
        with patch.object(extractor.cv2, "VideoWriter", return_value=writer):
            metadata = json.loads(extractor.process_row(
                pd.Series({"path": str(dicom)}), str(self.root), None, "path"
            ))
        self.assertEqual(metadata["error"], "Video encoding failed")
        self.assertNotIn("video_path", metadata)
        writer.write.assert_not_called()
        writer.release.assert_called_once()

    def test_empty_csv_preserves_original_exception(self):
        path = self.root / "empty.csv"
        path.touch()
        with self.assertRaises(ValueError) as raised:
            extractor.extract_h264_and_metadata(str(path))
        self.assertIsInstance(raised.exception.__cause__, pd.errors.EmptyDataError)

    def test_cli_comma_csv_exports_png_and_avi_with_metadata(self):
        screenshot = self.write_dicom(np.zeros((12, 20), dtype=np.uint8), name="screenshot")
        cine = self.write_dicom(np.zeros((5, 12, 20), dtype=np.uint8), name="cine", fps_tag=(0x0008, 0x2144))
        csv = self.root / "input.csv"
        pd.DataFrame({"path": [str(screenshot), str(cine)], "unused": [1, 2]}).to_csv(csv, index=False)
        result = subprocess.run([
            sys.executable, str(Path(extractor.__file__)),
            "--input_file", str(csv), "--sep", ",", "--file_type", "dicom",
            "--file_path_column", "path", "--output_dir", str(self.root / "exports"),
            "--metadata_dir", str(self.root / "metadata"), "--num_processes", "1",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = pd.read_csv(self.root / "metadata" / "input.csv_metadata_extracted.csv")
        self.assertEqual(set(metadata["FileName"]), {
            str(self.root / "exports" / "screenshot.png"),
            str(self.root / "exports" / "cine.avi"),
        })
        for path in metadata["FileName"]:
            self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
