"""Read-only rendering of saved eye masks and coordinates, with no inference or evaluation."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from animal_eye_metrology.errors import Phase1Error

BACKGROUND = (19, 26, 38)
PREDICTION = (50, 205, 237)
REFERENCE = (75, 255, 120)
POINT = (255, 225, 65)
BOX = (249, 120, 225)
ROI = (190, 199, 213)
RIGHT_EYE = (250, 150, 60)
BINOCULAR_LINE = (255, 255, 255)
RIGHT_PAIR_LINE = (240, 190, 255)


def _coordinates(
    value: Any, count: int, name: str, *, ordered_box: bool = False
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != count:
        raise Phase1Error(f"Invalid {name} coordinates for the result overlay.")
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise Phase1Error(f"Invalid {name} coordinates for the result overlay.") from error
    if not all(isfinite(item) for item in coordinates):
        raise Phase1Error(f"Non-finite {name} coordinates for the result overlay.")
    if ordered_box and (coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]):
        raise Phase1Error(f"Invalid {name} extent for the result overlay.")
    return coordinates


def _centroid(metrics: Mapping[str, Any], prefix: str) -> tuple[float, ...] | None:
    coordinates = (metrics.get(f"{prefix}_cx_px"), metrics.get(f"{prefix}_cy_px"))
    if all(value in (None, "") for value in coordinates):
        return None
    return _coordinates(coordinates, 2, f"{prefix} centroid")


def _binary_mask(mask: Image.Image, image_size: tuple[int, int], name: str) -> Image.Image:
    if mask.size != image_size:
        raise Phase1Error(f"{name} mask and original image dimensions differ.")
    if mask.mode not in {"1", "L", "I", "I;16"}:
        raise Phase1Error(f"{name} mask must contain a single channel.")
    # Existing stored masks use 0/255; accepting 0/1 also supports synthetic fixtures.
    return mask.convert("L").point(lambda value: 255 if value else 0)


def render_eye_comparison(
    image: Image.Image,
    prediction_mask: Image.Image | None,
    reference_mask: Image.Image,
    metrics: Mapping[str, Any],
    prediction: Mapping[str, Any],
    view: tuple[int, int, int, int],
    size: int = 480,
) -> Image.Image:
    """Render one eye in a caller-selected, shared original-image viewport.

    ``view`` uses integer pixel edges and must lie inside the original image.
    The saved masks are nearest-neighbor enlarged. Reference boundaries follow
    foreground pixel edges, while centroid coordinates already follow the
    project's (x + 0.5, y + 0.5) pixel-center convention and are used unchanged.
    No values are derived from masks, and input images/dictionaries are never
    modified. A missing prediction mask is allowed so failures remain viewable.

    The yellow cross represents only ``prediction['point_xy']``. In particular,
    an HRNet geometry center is not drawn as a SAM positive point for box-only
    conditions. Missing centroids are omitted rather than placed at (0, 0).
    """
    if not isinstance(size, int) or isinstance(size, bool) or size < 32:
        raise Phase1Error("Result overlay size must be an integer of at least 32 pixels.")
    if len(view) != 4 or any(
        not isinstance(value, int) or isinstance(value, bool) for value in view
    ):
        raise Phase1Error("Result overlay viewport must contain four integer pixel edges.")
    x0, y0, x1, y1 = view
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise Phase1Error("Result overlay viewport must be nonempty and inside the image.")
    reference = _binary_mask(reference_mask, image.size, "Reference")
    crop = image.convert("RGB").crop(view)
    if prediction_mask is not None:
        mask = _binary_mask(prediction_mask, image.size, "Prediction").crop(view)
        tint = Image.blend(crop, Image.new("RGB", crop.size, PREDICTION), alpha=0.55)
        crop = Image.composite(tint, crop, mask)

    scale = size / max(crop.size)
    width, height = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
    canvas = crop.resize((width, height), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    sx, sy = width / crop.width, height / crop.height

    def xy(x: float, y: float) -> tuple[float, float]:
        return (x - x0) * sx, (y - y0) * sy

    roi = _coordinates(prediction.get("roi_xyxy"), 4, "ROI", ordered_box=True)
    if roi is not None:
        draw.rectangle((*xy(roi[0], roi[1]), *xy(roi[2], roi[3])), outline=ROI, width=1)
    pixels = reference.load()
    # Draw true pixel-edge segments, rather than a rescaled, thick raster contour.
    # Neighbors are read in the complete mask so cropping creates no false edges.
    for y in range(y0, y1):
        for x in range(x0, x1):
            if not pixels[x, y]:
                continue
            for empty, start, end in (
                (y == 0 or not pixels[x, y - 1], (x, y), (x + 1, y)),
                (y + 1 == image.height or not pixels[x, y + 1], (x, y + 1), (x + 1, y + 1)),
                (x == 0 or not pixels[x - 1, y], (x, y), (x, y + 1)),
                (x + 1 == image.width or not pixels[x + 1, y], (x + 1, y), (x + 1, y + 1)),
            ):
                if empty:
                    draw.line((*xy(*start), *xy(*end)), fill=REFERENCE, width=2)
    box = _coordinates(prediction.get("box_xyxy"), 4, "prompt box", ordered_box=True)
    if box is not None:
        draw.rectangle((*xy(box[0], box[1]), *xy(box[2], box[3])), outline=BOX, width=2)
    reference_center = _centroid(metrics, "reference")
    if reference_center is not None:
        x, y = xy(*reference_center)
        draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline=REFERENCE, width=2)
    predicted_center = _centroid(metrics, "predicted")
    if predicted_center is not None:
        x, y = xy(*predicted_center)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=PREDICTION, width=2)
    point = _coordinates(prediction.get("point_xy"), 2, "SAM prompt point")
    if point is not None:
        x, y = xy(*point)
        draw.line((x - 6, y, x + 6, y), fill=POINT, width=2)
        draw.line((x, y - 6, x, y + 6), fill=POINT, width=2)

    result = Image.new("RGB", (size, size), BACKGROUND)
    # All overlays are clipped to the viewport before adding letterbox margins.
    result.paste(canvas, ((size - width) // 2, (size - height) // 2))
    return result


def render_measurement_overview(
    image: Image.Image,
    masks: Sequence[Image.Image | None],
    eyes: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
) -> Image.Image:
    """Overlay saved predicted centers and defined distance endpoints on a full image.

    ``masks`` and ``eyes`` must be in the same order. Missing masks are retained
    as missing; this function neither fills them nor recomputes measurements.
    The result retains the original dimensions and coordinate system. Callers
    must identify the condition and display its measurement qualification status.
    """
    if len(masks) != len(eyes):
        raise Phase1Error("Every eye result needs one corresponding mask entry.")
    canvas = image.convert("RGB").copy()
    for mask, eye in zip(masks, eyes, strict=True):
        side = eye.get("eye_side")
        if side not in {"left", "right"}:
            raise Phase1Error("Eye side must be left or right for the result overlay.")
        if mask is not None:
            binary = _binary_mask(mask, image.size, "Prediction")
            color = PREDICTION if side == "left" else RIGHT_EYE
            tint = Image.blend(canvas, Image.new("RGB", image.size, color), alpha=0.55)
            canvas = Image.composite(tint, canvas, binary)
    draw = ImageDraw.Draw(canvas)
    for row in measurements:
        distance = row.get("predicted_distance_px")
        if distance in (None, ""):
            continue
        try:
            valid_distance = isfinite(float(distance)) and float(distance) >= 0
        except (TypeError, ValueError):
            valid_distance = False
        if not valid_distance:
            raise Phase1Error("Invalid saved predicted distance for the result overlay.")
        endpoints = _coordinates(
            [
                row.get("predicted_cx_a_px"),
                row.get("predicted_cy_a_px"),
                row.get("predicted_cx_b_px"),
                row.get("predicted_cy_b_px"),
            ],
            4,
            "measurement endpoints",
        )
        kind = row.get("measurement_type")
        if kind not in {"binocular", "right_eye_pair"}:
            raise Phase1Error("Unknown measurement type for the result overlay.")
        draw.line(
            endpoints,
            fill=BINOCULAR_LINE if kind == "binocular" else RIGHT_PAIR_LINE,
            width=1,
        )
    font = ImageFont.load_default(size=12)
    for eye in eyes:
        center = _centroid(eye, "predicted")
        if center is None:
            continue
        x, y = center
        side = eye["eye_side"]
        color = PREDICTION if side == "left" else RIGHT_EYE
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
        label = f"{eye.get('annotation_id', '?')} {'L' if side == 'left' else 'R'}"
        draw.text((x + 4, y + 5 if side == "left" else y - 17), label, fill=color, font=font)
    return canvas
