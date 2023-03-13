"""Have fun cleaning virtual windows."""

import argparse
import fractions
import importlib.resources
import os
import pathlib
import sys
from typing import List, Optional, Final, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import pygame
import pygame.freetype
from icontract import require, ensure

import masterigayim

assert masterigayim.__doc__ == __doc__

PACKAGE_DIR = (
    pathlib.Path(str(importlib.resources.files(__package__)))
    if __package__ is not None
    else pathlib.Path(os.path.realpath(__file__)).parent
)


class Media:
    """Represent all the media in the file system."""

    def __init__(
        self,
        level_paths: List[pathlib.Path],
        font: pygame.freetype.Font,  # type: ignore
        game_over_song: pygame.mixer.Sound,
    ) -> None:
        """Initialize with the given values."""
        self.level_paths = level_paths
        self.font = font
        self.game_over_song = game_over_song


@ensure(lambda result: (result[0] is not None) ^ (result[1] is not None))
def initialize_media() -> Tuple[Optional[Media], Optional[str]]:
    """
    Collect a new media collection.

    Return the media collection, or an error if something could not be loaded.
    """
    images_dir = PACKAGE_DIR / "media/images"

    level_paths = []  # type: List[pathlib.Path]
    for pth in sorted(images_dir.glob("level*.png")):
        try:
            image = cv2.imread(str(pth))
        except Exception as exception:
            return None, str(exception)

        height, width, _ = image.shape
        if width != SCENE_WIDTH or height != SCENE_HEIGHT:
            return None, (
                f"Expected all levels to be 320x240, "
                f"but got an image {width}x{height}: {pth}"
            )

        level_paths.append(pth)

    # noinspection PyUnusedLocal
    font = None
    game_over_song = None  # type: Optional[pygame.mixer.Sound]

    try:
        font = pygame.freetype.Font(  # type: ignore
            str(PACKAGE_DIR / "media/fonts/freesansbold.ttf")
        )

        game_over_song = pygame.mixer.Sound(
            str(PACKAGE_DIR / "media/sfx/wooden_work.ogg")
        )

    except Exception as exception:
        return None, str(exception)

    assert font is not None

    return (
        Media(level_paths=level_paths, font=font, game_over_song=game_over_song),
        None,
    )


def cvmat_to_surface(image: cv2.Mat) -> pygame.surface.Surface:
    """Convert from OpenCV to pygame."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    return pygame.image.frombuffer(image_rgb.tobytes(), (width, height), "RGB")


class Duration:
    """Represent a time interval in seconds since epoch."""

    #: Start of a time interval.
    start: Final[float]

    #: End of a time interval.
    end: Final[float]

    def __init__(self, start: float, end: float) -> None:
        """Initialize with the given values."""
        self.start = start
        self.end = end


SCENE_WIDTH = 320
SCENE_HEIGHT = 240


def crop_and_resize_camera_frame_to_scene(frame_bgr: cv2.Mat) -> cv2.Mat:
    """Crop and resize the frame coming from the camera to the game scene."""
    frame_height, frame_width, _ = frame_bgr.shape

    expected_aspect_ratio = fractions.Fraction(SCENE_WIDTH, SCENE_HEIGHT)
    frame_aspect_ratio = fractions.Fraction(frame_width, frame_height)
    if frame_aspect_ratio != expected_aspect_ratio:
        if frame_height < frame_width:
            cropped_width = expected_aspect_ratio * frame_height
            center = int(frame_width / 2)
            left_from_center = max(0, round(center - cropped_width / 2))
            right_from_center = min(frame_width, round(center + cropped_width / 2))

            cropped = frame_bgr[..., left_from_center:right_from_center]
        else:
            cropped_height = frame_width / expected_aspect_ratio
            center = int(frame_height / 2)
            up_from_center = max(0, round(center - cropped_height / 2))
            down_from_center = min(frame_height, round(center + cropped_height / 2))

            cropped = frame_bgr[up_from_center:down_from_center, ...]

        return cv2.resize(cropped, (SCENE_WIDTH, SCENE_HEIGHT))
    else:
        if frame_width != SCENE_WIDTH or frame_height != SCENE_HEIGHT:
            return cv2.resize(frame_bgr, (SCENE_WIDTH, SCENE_HEIGHT))
        else:
            return frame_bgr.copy()


# Magnitude of optical flow to consider a sweep
MIN_MAGNITUDE_FOR_SWEEP = 7


class Level:
    """Represent a level in the game."""

    #: Original image of the level, in BGR
    image_bgr: Final[cv2.Mat]

    #: Mask of the surfaces which can be cleaned; either 0.0 or 1.0
    mask: Final[npt.NDArray[np.float64]]

    #: Dirt image, all pixels in [0, 1] range
    dirt: npt.NDArray[np.float64]

    #: Number of the original dirty pixels in the level
    total_dirty_pixels: Final[int]

    # fmt: off
    @require(
        lambda image_bgr:
        image_bgr.shape[1] == SCENE_WIDTH
        and image_bgr.shape[0] == SCENE_HEIGHT
    )
    @ensure(
        lambda self:
        self.image_bgr.shape[1] == self.mask.shape[1] == self.dirt.shape[1]
        and self.image_bgr.shape[0] == self.mask.shape[0] == self.dirt.shape[0]
    )
    @ensure(lambda self: (self.dirt[self.mask == 1.0] == 1.0).all())
    @ensure(lambda self: (self.dirt[self.mask == 0.0] == 0.0).all())
    @ensure(lambda self: self.total_dirty_pixels == np.sum(self.dirt))
    @ensure(lambda self: self.total_dirty_pixels == np.sum(self.mask))
    # fmt: on
    def __init__(self, image_bgr: npt.NDArray[np.uint8]) -> None:
        """Initialize with the given values."""
        self.image_bgr = image_bgr

        height, width, _ = image_bgr.shape

        mask = np.zeros((height, width), dtype=float)
        dirt = np.zeros((height, width), dtype=float)

        total_dirty_pixels = 0
        for row in range(height):
            for column in range(width):
                if (image_bgr[row, column] == (0, 0, 0)).all():
                    mask[row, column] = 1.0
                    dirt[row, column] = 1.0
                    total_dirty_pixels += 1

        self.mask = mask
        self.dirt = dirt
        self.total_dirty_pixels = total_dirty_pixels
        self.remaining_dirty_pixels = total_dirty_pixels

    # fmt: off
    @require(
        lambda self, optical_flow_magnitude:
        not (optical_flow_magnitude is not None)
        or (
            optical_flow_magnitude.shape[0] == self.dirt.shape[0]
            and optical_flow_magnitude.shape[1] == self.dirt.shape[1]
        )
    )
    @ensure(lambda self: self.remaining_dirty_pixels <= self.total_dirty_pixels)
    # fmt: on
    def update_dirt(self, optical_flow_magnitude: Optional[cv2.Mat]) -> None:
        """Update the state of the dirt in-situ given the optical flow."""
        if optical_flow_magnitude is not None:
            # Everything above 10 is considered swept. The magnitude range is
            # about [5, 90] for a fast moving hand.
            swept = np.zeros(optical_flow_magnitude.shape, dtype=float)
            swept[optical_flow_magnitude > MIN_MAGNITUDE_FOR_SWEEP] = CLEAN_DELTA
            swept = swept * self.mask

            self.dirt = np.clip(self.dirt - swept, 0, 1)

            # We clip here at some dirty level as the last dirt is almost invisible.
            self.dirt[self.dirt < 0.05] = 0

            self.remaining_dirty_pixels = np.sum(self.dirt > 0)  # type: ignore


def load_level(image_path: pathlib.Path) -> Level:
    """Load the level from the file system."""
    image = cv2.imread(str(image_path))
    return Level(image_bgr=image)


#: How long the level prelude takes, in seconds
PRELUDE_DURATION = 5


class State:
    """Represent the mutable state of the game."""

    now: float

    #: Index of the level, starting with 0
    level_id: int

    #: Current level
    level: Level

    #: If set, show the prelude for the next level
    prelude: Optional[Duration]

    #: Timestamp when the game started
    game_start: float

    #: Timestamp when the game finished
    game_over: Optional[float]

    def __init__(self, first_level: Level, now: float) -> None:
        """Initialize with the given values for the very first level."""
        self.now = now
        self.level_id = 0
        self.level = first_level
        self.prelude = Duration(now, now + PRELUDE_DURATION)
        self.game_start = now
        self.game_over = None


def initialize_state(media: Media, now: float) -> State:
    """Initialize the state for the first level."""
    pygame.mixer.stop()
    first_level = load_level(image_path=media.level_paths[0])
    return State(first_level=first_level, now=now)


class OpticalFlower:
    """Compute the optical flow continuously."""

    def __init__(self) -> None:
        # Previous frame, in grayscale
        self._previous_frame_gray = None  # type: Optional[cv2.Mat]

        self._magnitude = None  # type: Optional[cv2.Mat]
        self._angle = None  # type: Optional[cv2.Mat]

    # fmt: off
    @ensure(
        lambda self, result:
        (
            not (result is not None) or self.angle() is not None
        ) and (
            not (result is None) or self.angle() is None
        )
    )
    @ensure(
        lambda result:
        not (result is not None)
        or (
                result.shape[1] == SCENE_WIDTH and result.shape[0] == SCENE_HEIGHT
        )
    )
    # fmt: on
    def magnitude(self) -> Optional[cv2.Mat]:
        """
        Give the magnitude of the optical flow, if two frames have been observed.

        Sweeping magnitudes of a hand are roughly in the range [5, 90].
        """
        return self._magnitude

    # fmt: off
    @ensure(
        lambda self, result:
        (
            not (result is not None) or self.magnitude() is not None
        ) and (
            not (result is None) or self.magnitude() is None
        )
    )
    @ensure(
        lambda result:
        not (result is not None)
        or (
                result.shape[1] == SCENE_WIDTH and result.shape[0] == SCENE_HEIGHT
        )
    )
    # fmt: on
    def angle(self) -> Optional[cv2.Mat]:
        """Give the angle map of the optical flow, if two frames have been observed."""
        return self._magnitude

    # fmt: off
    @require(
        lambda frame_bgr:
        frame_bgr.shape[1] == SCENE_WIDTH and frame_bgr.shape[0] == SCENE_HEIGHT
    )
    # fmt: on
    def observe(self, frame_bgr: cv2.Mat) -> None:
        """Compute the optical flow from the previous frame based on this ``frame``."""
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._previous_frame_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._previous_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            self._magnitude, self._angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        self._previous_frame_gray = frame_gray


#: How much dirt to chip off at every sweep
CLEAN_DELTA = 0.05


def update_state(
    state: State, now: float, optical_flow_magnitude: Optional[cv2.Mat], media: Media
) -> None:
    """Update the state of the game on a tick event."""
    state.now = now

    if state.prelude is not None and now > state.prelude.end:
        state.prelude = None

    if state.game_over is not None:
        return

    # We stop at a portion of the dirty windows, so that people are not frustrated
    # with searching for the very last bits.
    if state.level.remaining_dirty_pixels < state.level.total_dirty_pixels * 0.1:
        if state.level_id == len(media.level_paths) - 1:
            pygame.mixer.stop()
            media.game_over_song.play()
            state.game_over = now
            return

        state.level_id += 1
        state.level = load_level(image_path=media.level_paths[state.level_id])
        state.prelude = Duration(now, now + PRELUDE_DURATION)
        return

    if state.game_over is None and state.prelude is None:
        state.level.update_dirt(optical_flow_magnitude)


# fmt: off
@require(lambda state: state.game_over is None)
@require(
    lambda state:
    state.prelude is None
    or state.now < state.prelude.start
    or state.now >= state.prelude.end
)
# fmt: on
def render_in_game(
    state: State,
    media: Media,
    frame_bgr: cv2.Mat,
    optical_flow_magnitude: Optional[cv2.Mat],
) -> pygame.surface.Surface:
    """Render the active game state as a scene."""
    not_mask = 1 - state.level.mask
    alpha = not_mask + state.level.dirt

    # Subtract 0.1 so that the frame still shows up a bit.
    alpha = np.clip(alpha - 0.1, 0, 1)

    one_minus_alpha = 1 - alpha

    background = frame_bgr.copy()
    if optical_flow_magnitude is not None:
        sweeps = np.logical_and(
            optical_flow_magnitude > MIN_MAGNITUDE_FOR_SWEEP, state.level.mask == 1
        )

        background[sweeps, 0] = 255
        background[sweeps, 1] = 255
        background[sweeps, 2] = 255

    combined = state.level.image_bgr.copy()
    combined[..., 0] = combined[..., 0] * alpha + background[..., 0] * one_minus_alpha
    combined[..., 1] = combined[..., 1] * alpha + background[..., 1] * one_minus_alpha
    combined[..., 2] = combined[..., 2] * alpha + background[..., 2] * one_minus_alpha

    assert combined.shape[0] == SCENE_HEIGHT and combined.shape[1] == SCENE_WIDTH
    scene = cvmat_to_surface(combined)

    percentage = round(
        (state.level.remaining_dirty_pixels / state.level.total_dirty_pixels) * 100
    )

    media.font.render_to(
        scene,
        (5, 5),
        f"Dirty pixels: {percentage}%",
        (0, 0, 0),
        size=10,
    )

    media.font.render_to(
        scene,
        (20, 220),
        'Press "q" to quit and "r" to restart',
        (0, 0, 0),
        size=10,
    )

    return scene


# fmt: off
@require(
    lambda state:
    state.prelude is not None and state.prelude.start <= state.now < state.prelude.end
)
# fmt: on
def render_prelude(state: State, media: Media) -> pygame.surface.Surface:
    """Render the prelude screen."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((0, 0, 0))

    assert state.prelude is not None
    remaining_seconds = round(state.prelude.end - state.now)

    media.font.render_to(
        scene,
        (5, 5),
        f"The level starts in: {remaining_seconds} seconds",
        (255, 255, 255),
        size=10,
    )

    media.font.render_to(
        scene,
        (20, 220),
        'Press "q" to quit and "r" to restart',
        (255, 255, 255),
        size=10,
    )

    return scene


@require(lambda state: state.game_over is not None)
def render_game_over(state: State, media: Media) -> pygame.surface.Surface:
    """Render the prelude screen."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((0, 0, 0))

    assert state.game_over is not None
    game_seconds = round(state.game_over - state.game_start)

    media.font.render_to(
        scene,
        (5, 5),
        f"You made it! Time: {game_seconds} seconds",
        (255, 255, 255),
        size=10,
    )

    media.font.render_to(
        scene,
        (20, 220),
        'Press "q" to quit and "r" to restart',
        (255, 255, 255),
        size=10,
    )

    return scene


def render(
    state: State, media: Media, frame_bgr: cv2.Mat, optical_flow_magnitude: cv2.Mat
) -> pygame.surface.Surface:
    """Render the game state as a scene."""
    if state.game_over is not None:
        return render_game_over(state, media)
    elif (
        state.prelude is not None
        and state.prelude.start <= state.now < state.prelude.end
    ):
        return render_prelude(state, media)
    else:
        return render_in_game(state, media, frame_bgr, optical_flow_magnitude)


def resize_image_to_canvas_and_blit(
    image: pygame.surface.Surface, canvas: pygame.surface.Surface
) -> None:
    """Draw the image on canvas resizing it to maximum at constant aspect ratio."""
    canvas.fill((0, 0, 0))

    canvas_aspect_ratio = fractions.Fraction(canvas.get_width(), canvas.get_height())
    image_aspect_ratio = fractions.Fraction(image.get_width(), image.get_height())

    if image_aspect_ratio < canvas_aspect_ratio:
        new_image_height = canvas.get_height()
        new_image_width = image.get_width() * (new_image_height / image.get_height())

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        margin = int((canvas.get_width() - image.get_width()) / 2)

        canvas.blit(image, (margin, 0))

    elif image_aspect_ratio == canvas_aspect_ratio:
        new_image_width = canvas.get_width()
        new_image_height = image.get_height()

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        canvas.blit(image, (0, 0))
    else:
        new_image_width = canvas.get_width()
        new_image_height = int(
            image.get_height() * (new_image_width / image.get_width())
        )

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        margin = int((canvas.get_height() - image.get_height()) / 2)

        canvas.blit(image, (0, margin))


def main(prog: str) -> int:
    """
    Execute the main routine.

    :param prog: name of the program to be displayed in the help
    :return: exit code
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--version", help="show the current version and exit", action="store_true"
    )

    # NOTE (mristin, 2022-12-23):
    # The module ``argparse`` is not flexible enough to understand special options such
    # as ``--version`` so we manually hard-wire.
    if "--version" in sys.argv and "--help" not in sys.argv:
        print(masterigayim.__version__)
        return 0

    parser.parse_args()

    pygame.init()
    pygame.mixer.pre_init()
    pygame.mixer.init()

    media, error = initialize_media()
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    assert media is not None

    pygame.display.set_caption("Master Igayim")
    surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    optical_flower = OpticalFlower()

    clock = pygame.time.Clock()

    state = initialize_state(media=media, now=pygame.time.get_ticks() / 1000)

    should_quit = False

    cap = None  # type: Optional[cv2.VideoCapture]
    try:
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and not should_quit:
            reading_ok, frame_bgr = cap.read()
            if not reading_ok:
                break

            # Flip so that it is easier for the player to understand the image
            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_bgr = crop_and_resize_camera_frame_to_scene(frame_bgr)

            optical_flower.observe(frame_bgr=frame_bgr)

            now = pygame.time.get_ticks() / 1000

            state.now = pygame.time.get_ticks() / 1000

            optical_flow_magnitude = optical_flower.magnitude()

            update_state(
                state,
                now=now,
                optical_flow_magnitude=optical_flow_magnitude,
                media=media,
            )

            scene = render(state, media, frame_bgr, optical_flow_magnitude)
            resize_image_to_canvas_and_blit(scene, surface)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_quit = True

                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    should_quit = True

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    state = initialize_state(media=media, now=now)
                else:
                    # Ignore all the other events
                    pass

            clock.tick(30)

    except Exception as exception:
        exc_type, _, exc_tb = sys.exc_info()
        assert (
            exc_tb is not None
        ), "Expected a traceback as we do not do anything fancy here"

        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        exc_type_name = getattr(exc_type, "__name__", None)
        if exc_type_name is None:
            exc_type_name = str(exc_type)

        print(
            f"Failed to process the video: "
            f"{exc_type_name} at {fname}:{exc_tb.tb_lineno} {exception}",
            file=sys.stderr,
        )
        return 1

    finally:
        if cap is not None:
            cap.release()

        print("Quitting...")
        pygame.quit()

    return 0


def entry_point() -> int:
    """Provide an entry point for a console script."""
    return main(prog="pop-that-balloon")


if __name__ == "__main__":
    sys.exit(main(prog="pop-that-balloon"))
