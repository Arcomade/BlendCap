"""Colour-correct frames OpenCV decoded with the wrong YUV formula.

A video stores brightness + two colour-difference channels, and names
the formula that turns them back into RGB (its "matrix": BT.601 for SD,
BT.709 for HD, BT.2020 for UHD/HDR). OpenCV's reader ignores that name
and always uses BT.601 (measured on the pinned opencv-python 4.13: a
BT.709 clip comes back up to 28/255 off on saturated colours, mean 5.5).
Blender honours it, and reads a file with no name as BT.601.

So an overlay written from OpenCV's frames (cv2.VideoWriter: BT.601, no
name in the file) showed shifted colours next to the original strip
whenever the source was tagged BT.709 - most HD camera and phone clips.

The fix is at the read: find the matrix the file names and convert the
frame from "decoded as BT.601" to "decoded as named". Both decodes are
linear in the same YUV values, so that is one 3x3 on the RGB values
(greys are untouched: every row sums to 1). The file written from the
corrected frames carries no name, is read back as BT.601 by OpenCV and
by Blender alike, and so needs nothing further.

The matrix is read from the file itself (no ffprobe in the capture
environment): MP4 / MOV / M4V containers holding H.264 or HEVC, from
the bitstream header and the container's `colr` box, resolved the way
FFmpeg resolves them (see _resolve). Anything else - MKV, AVI, other
codecs, an unreadable header - reports "not tagged" and the frames are
left exactly as before.

This is tied to OpenCV always decoding as BT.601. If a later OpenCV
starts honouring the tag, this correction must go.
"""
import struct

# matrix_coefficients code (ISO/IEC 23091-2, same numbers FFmpeg uses)
# -> luma weights (Kr, Kb). Only the ones libswscale has a table for;
# it treats every other code as BT.601, which is what OpenCV did anyway.
_KR_KB = {
    1: (0.2126, 0.0722),    # BT.709
    4: (0.30, 0.11),        # FCC
    5: (0.299, 0.114),      # BT.470BG (BT.601 625)
    6: (0.299, 0.114),      # SMPTE 170M (BT.601 525)
    7: (0.212, 0.087),      # SMPTE 240M
    9: (0.2627, 0.0593),    # BT.2020 non-constant
    10: (0.2627, 0.0593),   # BT.2020 constant (swscale uses the same table)
}
_OPENCV_KR_KB = (0.299, 0.114)
_NAMES = {1: "BT.709", 4: "FCC", 5: "BT.601", 6: "BT.601", 7: "SMPTE 240M",
          9: "BT.2020", 10: "BT.2020"}

_MAX_MOOV = 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Bit reader for H.264 / HEVC parameter sets
# ---------------------------------------------------------------------------

class _Bits:
    def __init__(self, nal_payload):
        # Drop emulation-prevention bytes (00 00 03 -> 00 00).
        out = bytearray()
        zeros = 0
        for b in nal_payload:
            if zeros >= 2 and b == 3:
                zeros = 0
                continue
            out.append(b)
            zeros = zeros + 1 if b == 0 else 0
        self.data = bytes(out)
        self.pos = 0

    def u(self, n):
        v = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]      # IndexError = ran off the end
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def skip(self, n):
        self.pos += n
        if self.pos > len(self.data) * 8:
            raise IndexError("past end")

    def ue(self):
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 32:
                raise ValueError("bad exp-golomb")
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def _vui_matrix(b):
    """Shared start of the H.264 and HEVC VUI: the matrix code, or None
    when the stream gives no colour description."""
    if b.u(1):                      # aspect_ratio_info_present_flag
        if b.u(8) == 255:           # Extended_SAR
            b.skip(32)
    if b.u(1):                      # overscan_info_present_flag
        b.skip(1)
    if not b.u(1):                  # video_signal_type_present_flag
        return None
    b.skip(4)                       # video_format(3) + video_full_range_flag
    if not b.u(1):                  # colour_description_present_flag
        return None
    b.skip(16)                      # colour_primaries, transfer_characteristics
    return b.u(8)                   # matrix_coefficients


def _h264_sps_matrix(nal):
    b = _Bits(nal[1:])              # 1-byte NAL header
    profile = b.u(8)
    b.skip(16)                      # constraint flags + level_idc
    b.ue()                          # seq_parameter_set_id
    if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        chroma = b.ue()
        if chroma == 3:
            b.skip(1)               # separate_colour_plane_flag
        b.ue()                      # bit_depth_luma_minus8
        b.ue()                      # bit_depth_chroma_minus8
        b.skip(1)                   # qpprime_y_zero_transform_bypass_flag
        if b.u(1):                  # seq_scaling_matrix_present_flag
            for i in range(8 if chroma != 3 else 12):
                if b.u(1):          # seq_scaling_list_present_flag
                    last = nxt = 8
                    for _ in range(16 if i < 6 else 64):
                        if nxt != 0:
                            nxt = (last + b.se() + 256) % 256
                        last = last if nxt == 0 else nxt
    b.ue()                          # log2_max_frame_num_minus4
    poc_type = b.ue()
    if poc_type == 0:
        b.ue()                      # log2_max_pic_order_cnt_lsb_minus4
    elif poc_type == 1:
        b.skip(1)                   # delta_pic_order_always_zero_flag
        b.se()                      # offset_for_non_ref_pic
        b.se()                      # offset_for_top_to_bottom_field
        for _ in range(b.ue()):     # num_ref_frames_in_pic_order_cnt_cycle
            b.se()
    b.ue()                          # max_num_ref_frames
    b.skip(1)                       # gaps_in_frame_num_value_allowed_flag
    b.ue()                          # pic_width_in_mbs_minus1
    b.ue()                          # pic_height_in_map_units_minus1
    if not b.u(1):                  # frame_mbs_only_flag
        b.skip(1)                   # mb_adaptive_frame_field_flag
    b.skip(1)                       # direct_8x8_inference_flag
    if b.u(1):                      # frame_cropping_flag
        for _ in range(4):
            b.ue()
    if not b.u(1):                  # vui_parameters_present_flag
        return None
    return _vui_matrix(b)


def _hevc_sps_matrix(nal):
    b = _Bits(nal[2:])              # 2-byte NAL header
    b.skip(4)                       # sps_video_parameter_set_id
    max_sub = b.u(3)                # sps_max_sub_layers_minus1
    b.skip(1)                       # sps_temporal_id_nesting_flag
    # profile_tier_level
    b.skip(88 + 8)                  # general profile/tier/flags + level
    present = [(b.u(1), b.u(1)) for _ in range(max_sub)]
    if max_sub > 0:
        b.skip(2 * (8 - max_sub))
    for prof, lvl in present:
        if prof:
            b.skip(88)
        if lvl:
            b.skip(8)
    b.ue()                          # sps_seq_parameter_set_id
    if b.ue() == 3:                 # chroma_format_idc
        b.skip(1)                   # separate_colour_plane_flag
    b.ue()                          # pic_width_in_luma_samples
    b.ue()                          # pic_height_in_luma_samples
    if b.u(1):                      # conformance_window_flag
        for _ in range(4):
            b.ue()
    b.ue()                          # bit_depth_luma_minus8
    b.ue()                          # bit_depth_chroma_minus8
    log2_poc = b.ue() + 4           # log2_max_pic_order_cnt_lsb_minus4
    ordering = b.u(1)               # sps_sub_layer_ordering_info_present_flag
    for _ in range((max_sub + 1) if ordering else 1):
        b.ue(); b.ue(); b.ue()
    for _ in range(6):              # block / transform size + depth fields
        b.ue()
    if b.u(1):                      # scaling_list_enabled_flag
        if b.u(1):                  # sps_scaling_list_data_present_flag
            for size_id in range(4):
                for _ in range(0, 6, 3 if size_id == 3 else 1):
                    if not b.u(1):  # scaling_list_pred_mode_flag
                        b.ue()      # scaling_list_pred_matrix_id_delta
                    else:
                        if size_id > 1:
                            b.se()  # scaling_list_dc_coef_minus8
                        for _ in range(min(64, 1 << (4 + (size_id << 1)))):
                            b.se()  # scaling_list_delta_coef
    b.skip(2)                       # amp_enabled, sample_adaptive_offset
    if b.u(1):                      # pcm_enabled_flag
        b.skip(8)                   # pcm bit depths
        b.ue(); b.ue()
        b.skip(1)                   # pcm_loop_filter_disabled_flag
    # Short-term reference picture sets. Only each set's entry count is
    # needed to keep in step (counted the way FFmpeg counts them).
    counts = []
    for idx in range(b.ue()):       # num_short_term_ref_pic_sets
        if idx and b.u(1):          # inter_ref_pic_set_prediction_flag
            b.skip(1)               # delta_rps_sign
            b.ue()                  # abs_delta_rps_minus1
            n = 0
            for _ in range(counts[idx - 1] + 1):
                used = b.u(1)       # used_by_curr_pic_flag
                if used or b.u(1):  # use_delta_flag
                    n += 1
            counts.append(n)
        else:
            neg, pos = b.ue(), b.ue()
            for _ in range(neg + pos):
                b.ue()              # delta_poc_s*_minus1
                b.skip(1)           # used_by_curr_pic_s*_flag
            counts.append(neg + pos)
    if b.u(1):                      # long_term_ref_pics_present_flag
        for _ in range(b.ue()):     # num_long_term_ref_pics_sps
            b.skip(log2_poc + 1)
    b.skip(2)                       # temporal_mvp, strong_intra_smoothing
    if not b.u(1):                  # vui_parameters_present_flag
        return None
    return _vui_matrix(b)


# ---------------------------------------------------------------------------
# MP4 / MOV container
# ---------------------------------------------------------------------------

def _boxes(buf, start, end):
    """Yield (type, payload_start, payload_end) for the boxes in buf[start:end]."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", buf, pos)
        head = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            return
        yield kind, pos + head, pos + size
        pos += size


def _find(buf, start, end, kind):
    for k, s, e in _boxes(buf, start, end):
        if k == kind:
            return s, e
    return None


def _read_moov(path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        pos = 0
        seen = 0
        while pos + 8 <= file_size and seen < 4096:
            seen += 1
            f.seek(pos)
            head = f.read(16)
            if len(head) < 8:
                return None
            size, kind = struct.unpack_from(">I4s", head, 0)
            hlen = 8
            if size == 1:
                if len(head) < 16:
                    return None
                size = struct.unpack_from(">Q", head, 8)[0]
                hlen = 16
            elif size == 0:
                size = file_size - pos
            if size < hlen:
                return None
            if kind == b"moov":
                if size - hlen > _MAX_MOOV:
                    return None
                f.seek(pos + hlen)
                return f.read(size - hlen)
            pos += size
    return None


def _video_sample_entry(moov):
    """(codec fourcc, children_start, children_end) of the first video
    track's sample description, or None."""
    for kind, s, e in _boxes(moov, 0, len(moov)):
        if kind != b"trak":
            continue
        mdia = _find(moov, s, e, b"mdia")
        if not mdia:
            continue
        hdlr = _find(moov, mdia[0], mdia[1], b"hdlr")
        if not hdlr or moov[hdlr[0] + 8:hdlr[0] + 12] != b"vide":
            continue
        minf = _find(moov, mdia[0], mdia[1], b"minf")
        stbl = _find(moov, minf[0], minf[1], b"stbl") if minf else None
        stsd = _find(moov, stbl[0], stbl[1], b"stsd") if stbl else None
        if not stsd:
            continue
        for k, es, ee in _boxes(moov, stsd[0] + 8, stsd[1]):
            # VisualSampleEntry: 78 bytes of fixed fields, then child boxes
            return k, es + 78, ee
    return None


_H264 = (b"avc1", b"avc2", b"avc3", b"avc4")
_HEVC = (b"hvc1", b"hev1", b"dvh1", b"dvhe")
_UNREAD = object()      # no parameter set in the header, or one this can't follow


def _resolve(codec, stream, colr):
    """Which value FFmpeg ends up with (checked against ffprobe):

      H.264  the bitstream's colour description when it has one - even
             one that says "unspecified" - else the container's `colr`.
      HEVC   the bitstream only; its decoder resets to "unspecified"
             when the stream has no description, ignoring `colr`.

    Other codecs carry it in places this doesn't read: say nothing."""
    if stream is _UNREAD:
        return None
    if codec in _HEVC:
        code = stream
    elif codec in _H264:
        code = stream if stream is not None else colr
    else:
        return None
    if not isinstance(code, int) or code == 2:      # 2 = unspecified
        return None
    return code


def read_matrix_code(path):
    """The matrix_coefficients code FFmpeg resolves for the file's first
    video track, or None when there is none (or the file isn't an
    MP4/MOV with H.264 or HEVC, or its headers can't be read)."""
    try:
        moov = _read_moov(str(path))
        if not moov:
            return None
        entry = _video_sample_entry(moov)
        if not entry:
            return None
        codec, start, end = entry
        colr = None
        stream = _UNREAD
        for kind, s, e in _boxes(moov, start, end):
            try:
                if kind == b"colr" and colr is None:
                    if moov[s:s + 4] in (b"nclx", b"nclc") and e - s >= 10:
                        colr = struct.unpack_from(">H", moov, s + 8)[0]
                elif kind == b"avcC" and stream is _UNREAD:
                    if moov[s + 5] & 0x1F:
                        ln = struct.unpack_from(">H", moov, s + 6)[0]
                        stream = _h264_sps_matrix(moov[s + 8:s + 8 + ln])
                elif kind == b"hvcC" and stream is _UNREAD:
                    p = s + 23
                    for _ in range(moov[s + 22]):
                        nal_type = moov[p] & 0x3F
                        count = struct.unpack_from(">H", moov, p + 1)[0]
                        p += 3
                        for _i in range(count):
                            ln = struct.unpack_from(">H", moov, p)[0]
                            if nal_type == 33 and stream is _UNREAD:
                                stream = _hevc_sps_matrix(moov[p + 2:p + 2 + ln])
                            p += 2 + ln
            except Exception:
                # A parameter set this reader can't follow: say nothing
                # rather than guess from the container.
                stream = _UNREAD
        return _resolve(codec, stream, colr)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------

def _yuv_to_rgb(kr, kb):
    import numpy as np
    kg = 1.0 - kr - kb
    return np.array([
        [1.0, 0.0, 2.0 * (1.0 - kr)],
        [1.0, -2.0 * (1.0 - kb) * kb / kg, -2.0 * (1.0 - kr) * kr / kg],
        [1.0, 2.0 * (1.0 - kb), 0.0],
    ])


def correction_for_code(code):
    """3x3 for cv2.transform on a BGR frame OpenCV decoded from a file
    tagged with `code`, or None when no correction applies."""
    weights = _KR_KB.get(code)
    if weights is None or weights == _OPENCV_KR_KB:
        return None
    import numpy as np
    rgb = _yuv_to_rgb(*weights) @ np.linalg.inv(_yuv_to_rgb(*_OPENCV_KR_KB))
    return np.ascontiguousarray(rgb[::-1, ::-1])        # RGB order -> BGR


def frame_fix(path, log=None):
    """The correction for frames cv2.VideoCapture reads from `path`:
    a 3x3 for apply_fix, or None when the frames are right as they are.
    `log`, when given, gets one line saying what was found."""
    code = read_matrix_code(path)
    fix = correction_for_code(code)
    if log is not None and fix is not None:
        log(f"  Colour: source is tagged {_NAMES.get(code, code)}, "
            f"correcting the decoded frames to match.")
    return fix


def apply_fix(frame, fix):
    """Apply frame_fix's result to one BGR uint8 frame (new array), or
    return the frame untouched when there is nothing to fix."""
    if fix is None or frame is None:
        return frame
    import cv2
    return cv2.transform(frame, fix)
