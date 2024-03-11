import os
import pathlib

from typing import Optional

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.collections import LineCollection


class PlayerArtist(mpatches.Circle):
    """
    Artist to represent a player as a labelled disk.

    Subclasses Circle for most of its functionality.
    """

    def __init__(
        self,
        xy: tuple[float, float],
        *args,
        font_size: int = 10,
        **kwargs,
    ):
        """Setup the player artist.

        The artist is a labelled disk with the label centred on the
        disk.  The initialiser passes most arguments to the Circle
        superclass, but includes basic control of the text to be
        displayed.

        Args:
            xy (tuple[float, float]): Location of player.
            *args: Arguments passed to Circle.
            font_size (int): Font size of label.
            **kwargs: Keyword arguments passed to Circle.
        """
        self.text = Text(*xy, ha="center", va="center")
        super().__init__(xy, *args, **kwargs)

        self.text.set_label(self.get_label())
        self.text.set_text(self.get_label())
        self.text.set_fontsize(font_size)

        if self.get_fill():
            self.text.set_color("white")
        else:
            self.text.set_color(self.get_edgecolor())

    def set_figure(self, figure):
        """Set figure for artist and artist properties."""
        self.text.set_figure(figure)
        super().set_figure(figure)

    @mpatches.Circle.axes.setter
    def axes(self, new_axes):
        """Override the axes property setter to set Axes on child artists as well."""
        self.text.axes = new_axes
        mpatches.Circle.axes.fset(self, new_axes)  # Call the superclass property setter.

    def set_transform(self, transform):
        """Set the transform for the player and their label."""
        self.text.set_transform(transform)
        super().set_transform(transform)

    def set_center(self, xy):
        """Update the location of the disk and the label."""
        self.text.set_position(xy)
        super().set_center(xy)

    def draw(self, renderer):
        """Draw the player and their label."""
        super().draw(renderer)
        self.text.draw(renderer)

    def set_visible(self, visible):
        """Set the visibility of the player and their label."""
        self.text.set_visible(visible)
        super().set_visible(visible)

    def set_fontsize(self, size):
        """Set the font size of the label."""
        self.text.set_fontsize(size)

    def get_fontsize(self):
        """Get the font size of the label."""
        return self.text.get_fontsize()


class PlayerArtistWithTrace(PlayerArtist):
    """Artist to represent a player as a labelled disk with a trace."""

    def __init__(
        self,
        *args,
        trace_data: Optional[np.ndarray | list[list[tuple[int, int]]]] = None,
        max_trace_length: Optional[int] = None,
        **kwargs,
    ):
        """Setup the player artist.

        Args:
            *args: Arguments passed to PlayerArtist.
            trace_data (np.ndarray | list[list[tuple[int, int]]]): Trace data to plot.
            max_trace_length (int): Maximum length of trace to plot.
            **kwargs: Keyword arguments passed to PlayerArtist.
        """
        super().__init__(*args, **kwargs)

        if trace_data is None:
            trace_data = [self.get_center(), self.get_center()]
        trace_data = np.array(trace_data)

        self.max_trace_length = max_trace_length if max_trace_length is not None else len(trace_data)

        segments = np.stack([trace_data[:-1], trace_data[1:]], axis=1)[-self.max_trace_length :]

        trace_colors = [(*self.get_edgecolor()[:3], alpha) for alpha in np.linspace(0.1, 1, len(segments))]

        self.trace = LineCollection(segments=segments, colors=trace_colors, linewidths=2.0, zorder=self.zorder - 1)

    def set_trace(self, trace_data: np.ndarray | list[tuple[int, int]]):
        """Set the trace data for the player from (N * 2)-dimensional trajectory.

        This will update the trace, and the location of the player to
        the last location in the trace.

        Args:
            trace_data (np.ndarray | list[tuple[int, int]]): Trace data to plot.

        """

        trace_data = np.array(trace_data)
        segments = np.stack([trace_data[:-1], trace_data[1:]], axis=1)

        self.trace.set_segments(segments[-self.max_trace_length :])
        self.set_center(trace_data[-1])

    def set_figure(self, figure):
        """Set figure for artist."""
        super().set_figure(figure)
        self.trace.set_figure(figure)

    @LineCollection.axes.setter
    def axes(self, new_axes):
        """Override the axes property setter to set Axes on child artists as well."""
        self.trace.axes = new_axes
        LineCollection.axes.fset(self, new_axes)

    def set_transform(self, transform):
        """Set the transform for the player and their trace."""
        self.trace.set_transform(transform)
        super().set_transform(transform)

    def set_center(self, xy):
        """Update the location and the trace"""
        if len(self.trace.get_segments()) == 0:
            self.trace.set_segments(
                np.concatenate([[[xy, xy]]])[-self.max_trace_length :]
            )
        else:
            self.trace.set_segments(
                np.concatenate([self.trace.get_segments(), [[self.get_center(), xy]]])[-self.max_trace_length :]
            )

        self.trace.set_colors(
            [(*self.get_edgecolor()[:3], alpha) for alpha in np.linspace(0.1, 1, len(self.trace.get_segments()))]
        )

        super().set_center(xy)

    def draw(self, renderer):
        """Draw the player and their trace."""
        self.trace.draw(renderer)
        super().draw(renderer)

    def set_visible(self, visible):
        """Set the visibility of the player and their trace."""
        self.trace.set_visible(visible)
        if not visible:
            self.trace.set_segments(
                []
            )
        super().set_visible(visible)


class PlayerArtistWithVelocity(PlayerArtist):
    """Artist to represent a player as a labelled disk with velocity vector as arrow."""

    def __init__(
        self,
        *args,
        velocity: Optional[tuple[float, float]] = (0, 0),
        arrow_width: Optional[float] = 0.5,
        arrow_scale: Optional[float] = 2.0,
        auto_speed_update: bool = False,
        **kwargs,
    ):
        """Setup the player artist.

        The artist is a labelled disk with an arrow representing the
        velocity vector.  The length of the arrow is proportional to
        the magnitude of the velocity vector.

        Args:
            *args: Arguments passed to PlayerArtist.
            velocity (tuple[float, float]): Velocity vector of player.
            arrow_width (float): Width of the arrow.
            arrow_scale (float): Scale factor for the length of the arrow.
            **kwargs: Keyword arguments passed to PlayerArtist.

        """
        super().__init__(*args, **kwargs)

        self.auto_speed_update = auto_speed_update
        self.velocity = np.array(velocity)
        self.arrow_scale = np.float32(arrow_scale)

        self.arrow = mpatches.FancyArrow(
            *self._arrow_anchor,
            *self._arrow_length,
            width=arrow_width,
            ec=self.get_edgecolor(),
            fc=self.get_facecolor(),
            length_includes_head=False,
            zorder=self.zorder - 1,
        )

    @property
    def _arrow_anchor(self) -> tuple[float, float]:
        """Return the tail of the arrow.

        The arrow is positioned on the edge of the player disk."""
        velocity = self.velocity if np.linalg.norm(self.velocity) > 0 else np.array([0, 1])
        return self.get_center() + self.get_radius() * velocity / np.linalg.norm(velocity)

    @property
    def _arrow_length(self) -> tuple[float, float]:
        """Return the length of the arrow.

        The length of the arrow is proportional to the magnitude of
        the velocity vector, and scaled by the
        ``velocity_coefficient`` to an appropriate length relative to
        the radius of the marker.

        """
        if np.linalg.norm(self.velocity) == 0:
            return np.array([0, 0])
        return self.velocity * self.arrow_scale

    def set_figure(self, figure):
        """Set figure for artist."""
        super().set_figure(figure)
        self.arrow.set_figure(figure)

    @mpatches.FancyArrow.axes.setter
    def axes(self, new_axes):
        """Override the axes property setter to set Axes on child artists as well."""
        self.arrow.axes = new_axes
        mpatches.FancyArrow.axes.fset(self, new_axes)

    def set_transform(self, transform):
        """Set the transform for the player and their arrow."""
        self.arrow.set_transform(transform)
        super().set_transform(transform)

    def set_center(self, xy):
        """Update the location and the arrow."""
        tx, ty = [p - q for p, q in zip(xy, self.get_center())]
        super().set_center(xy)

        arrow_xy = mtransforms.Affine2D().translate(tx, ty).transform(self.arrow.get_xy())
        self.arrow.set_xy(arrow_xy)

        if self.auto_speed_update:
            self.set_velocity((tx, ty))

    def set_velocity(self, velocity):
        """Set the velocity of the playern and update arrow."""
        self.velocity = np.array(velocity)

        temp_arrow = mpatches.FancyArrow(
            *self._arrow_anchor,
            *self._arrow_length,
            width=self.arrow._width,
            length_includes_head=False,
        )

        self.arrow.set_xy(temp_arrow.get_xy())

    def draw(self, renderer):
        """Draw the player and their arrow."""
        self.arrow.draw(renderer)
        super().draw(renderer)

    def set_visible(self, visible):
        """Set the visibility of the player and their arrow."""
        self.arrow.set_visible(visible)
        super().set_visible(visible)


class TrajectoryArtist(Line2D):
    """Artist to represent a trajectory as a line."""

    pass


class PitchAxes(matplotlib.axes.Axes):
    """Customised Axes class for drawing sports pitches."""

    def __init__(self, *args, length: float = 105, width: float = 68, margin: float = 2, aspect="equal", **kwargs):
        """Set axes with markings of a sports pitch with origin at centre of pitch.

        Args:
            *args: Arguments passed to matplotlib.axes.Axes.
            length (float): Length of pitch in metres.
            width (float): Width of pitch in metres.
            margin (float): Margin around pitch in metres.
            aspect (str): Aspect ratio of pitch.
            **kwargs: Keyword arguments passed to matplotlib.axes.Axes.
        """
        super().__init__(
            *args,
            xlim=(-(length / 2 + margin), length / 2 + margin),
            ylim=(-(width / 2 + margin), width / 2 + margin),
            aspect=aspect,
            xticklabels="",
            yticklabels="",
        )

        self.set_axis_off()

        self.length = length
        self.width = width
        self.margin = margin

        self.colormap = matplotlib.colormaps["Set1"]

    def draw_pitch(self, simplified=False):
        """Draw the pitch markings."""
        raise NotImplementedError

    def draw_player(self, *args, team_id: Optional[int] = None, player_class: type = PlayerArtist, **kwargs):
        """
        Draw a player using the player_class passed, and colored based on team_id.

        Args:
            *args: Arguments passed to :class:``player_class``.
            team_id (int): Team ID to color the player.
            player_class (type): Class to use to draw the player.
            **kwargs: Keyword arguments passed to :class:``player_class``.

        Returns:
            The player artist.
        """
        if team_id is not None:
            kwargs["fc"] = self.colormap.colors[team_id]
            kwargs["ec"] = self.colormap.colors[team_id]

        player = player_class(*args, **kwargs)
        self.add_artist(player)

        return player

    def draw_trajectory(self, *args, team_id: Optional[int] = None, **kwargs):
        """Draw a trajectory using the TrajectoryArtist class, and colored based on team_id.

        Args:
            *args: Arguments passed to TrajectoryArtist.
            team_id (int): Team ID to color the trajectory.
            **kwargs: Keyword arguments passed to TrajectoryArtist.

        Returns:
            The trajectory artist.
        """
        if team_id is not None:
            kwargs["color"] = self.colormap.colors[team_id]

        trajectory = TrajectoryArtist(*args, **kwargs)
        self.add_line(trajectory)

        return trajectory


class SoccerAxes(PitchAxes):
    """Render a soccer pitch."""

    def __init__(self, *args, length: float = 105, width: float = 68, margin=5, **kwargs):
        """Setup ``Axes`` to render a soccer pitch.

        The pitch is rendered with the origin at the centre of the
        pitch, and the data coordinate system is in metres.

        Args:
            *args: Arguments passed to PitchAxes.
            length (float): Length of pitch in metres.
            width (float): Width of pitch in metres.
            margin (float): Margin around pitch in metres.
            **kwargs: Keyword arguments passed to PitchAxes.

        """

        # Add 2m margin around pitch.
        super().__init__(*args, length=length, width=width, margin=margin, **kwargs)

    def draw_pitch(self, simplified=False):
        """
        Draw the pitch markings.

        If the argument simplified is true then only draw the basic markings.

        Args:
            simplified (bool): If true, only draw the basic markings.
        """
        marking_args = {"color": "k", "zorder": 1}

        # Convenience method for drawing boxes.
        def draw_box(x_bl, y_bl, x_tr, y_tr):
            return mpatches.Rectangle((x_bl, y_bl), x_tr - x_bl, y_tr - y_bl, fill=False, **marking_args)

        # Draw pitch outline.
        self.add_patch(draw_box(-self.length / 2, -self.width / 2, self.length / 2, self.width / 2))

        # Penalty boxes.
        self.add_patch(draw_box(-self.length / 2, -20.15, -self.length / 2 + 16.50, 20.15))
        self.add_patch(draw_box(self.length / 2, -20.15, self.length / 2 - 16.50, 20.15))

        # 6-yard boxes.
        self.add_patch(draw_box(-self.length / 2, -9.15, -self.length / 2 + 5.50, 9.15))
        self.add_patch(draw_box(self.length / 2, -9.15, self.length / 2 - 5.50, 9.15))

        # Penalty spots.
        self.add_patch(mpatches.Circle((-self.length / 2 + 11, 0), radius=0.5, fill=True, **marking_args))
        self.add_patch(mpatches.Circle((self.length / 2 - 11, 0), radius=0.5, fill=True, **marking_args))

        # The Ds.
        self.add_patch(mpatches.Arc((-self.length / 2 + 11, 0), width=20, height=20, angle=0, theta1=305, theta2=55))
        self.add_patch(mpatches.Arc((self.length / 2 - 11, 0), width=20, height=20, angle=0, theta1=125, theta2=235))
        # Centre line and circle.
        self.add_line(Line2D((0, 0), (-self.width / 2, self.width / 2), **marking_args))

        self.add_patch(mpatches.Circle((0, 0), radius=9.15, fill=False, **marking_args))
        self.add_patch(mpatches.Circle((0, 0), radius=0.5, fill=True, **marking_args))

        # Corner arcs.
        self.add_patch(
            mpatches.Arc((-self.length / 2, -self.width / 2), width=2, height=2, angle=0, theta1=0, theta2=90)
        )
        self.add_patch(
            mpatches.Arc((self.length / 2, -self.width / 2), width=2, height=2, angle=0, theta1=90, theta2=180)
        )
        self.add_patch(
            mpatches.Arc((self.length / 2, self.width / 2), width=2, height=2, angle=0, theta1=180, theta2=270)
        )
        self.add_patch(
            mpatches.Arc((-self.length / 2, self.width / 2), width=2, height=2, angle=0, theta1=270, theta2=360)
        )


class FootballAxes(PitchAxes):
    """Render an American football field."""

    def __init__(self, *args, length: float = 120.0, width: float = 53.5, margin=5, **kwargs):
        """Setup the axes for a football field.

        Args:
            *args: Arguments passed to PitchAxes.
            length (float): Length of pitch in yards.
            width (float): Width of pitch in yards.
            margin (float): Margin around pitch in yards.
            **kwargs: Keyword arguments passed to PitchAxes.
        """
        super().__init__(*args, length=length, width=width, margin=margin, **kwargs)

    def draw_pitch(self, simplified=False):
        """
        Draw the pitch markings.
        
        If the argument simplified is true then only draw the basic markings.
        """
        marking_args = {"color": "#00AE42", "zorder": 1}

        # Convenience method for drawing boxes.
        def draw_box(x_bl, y_bl, x_tr, y_tr):
            return mpatches.Rectangle((x_bl, y_bl), x_tr - x_bl, y_tr - y_bl, fill=False, **marking_args)

        # Draw touch lines.
        self.add_patch(draw_box(-self.length / 2, -self.width / 2, self.length / 2, self.width / 2))

        # Draw the end-zones.
        self.add_patch(draw_box(-self.length / 2, -self.width / 2, -self.length / 2 + 10, self.width / 2))
        self.add_patch(draw_box(self.length / 2 - 10, -self.width / 2, self.length / 2, self.width / 2))

        # Draw the yard-lines.
        for x in range(-45, 50, 5):
            self.add_line(Line2D([x, x], [-self.width / 2, self.width / 2], **marking_args))

        # Draw hash-marks.
        for x in range(-49, 50, 1):
            self.add_line(Line2D([x, x], [25, 25.65], **marking_args))
            self.add_line(Line2D([x, x], [3.1, 3.8], **marking_args))
            self.add_line(Line2D([x, x], [-3.8, -3.1], **marking_args))
            self.add_line(Line2D([x, x], [-25.65, -25], **marking_args))

    def image_pitch(self, fname=None):
        """Use image background for pitch"""

        # Load the image data from file.
        if fname is None:
            fname = (
                pathlib.Path(os.getenv("DATA_DIR")) / "football" / "images" / "nfl" / "templates" / "football_field.png"
            )

        with open(fname, "rb") as f:
            T = plt.imread(f, format="png")

        self.imshow(T, extent=[-60, 60, -26.75, 26.75])

    def draw_line_marker(self, x, color="#ffc629", zorder=2):
        """ Draw first down marker or line-of-scrimmage marker. """
        self.add_line(Line2D([x, x], self.ylim, color=color))

    def draw_ball(self, x, y, radius=1, color="#a52a2a", zorder=200):
        """ Draw the ball at (x, y). """
        c = mpatches.Ellipse((x, y), width=radius * 2, height=radius, color=color, zorder=zorder)
        self.add_patch(c)
        return c


class HockeyAxes(PitchAxes):
    """ Draw North American hockey-specific ice-hockey rink markings in feet. """

    def __init__(self, *args, length: float = 200.0, width: float = 85.0, margin=5, **kwargs):
        """Setup the axes for a North American ice rink.

        Args:
            *args: Arguments passed to PitchAxes.
            length (float): Length of pitch in feet.
            width (float): Width of pitch in feet.
            margin (float): Margin around pitch in feet.
            **kwargs: Keyword arguments passed to PitchAxes.
        """
        super().__init__(*args, length=length, width=width, margin=margin, **kwargs)

    def draw_pitch(self, simplified=False):
        """
        Draw the pitch markings.
        
        If the argument simplified is true then only draw the basic markings.
        """
        marking_args = {"color": "k", "zorder": 1}

        # Draw Boards.
        for x_args, y_args in [
            ((-72, 72), (42.5, 42.5)),
            ((-72, 72), (-42.5, -42.5)),
            ((-100, -100), (-14.5, 14.5)),
            ((100, 100), (-14.5, 14.5)),
        ]:
            self.add_line(Line2D(x_args, y_args, **marking_args))

        for xy, angle in [((72, 14.5), 0), ((-72, 14.5), 90), ((-72, -14.5), 180), ((72, -14.5), 270)]:
            self.add_patch(mpatches.Arc(xy, width=56, height=56, angle=angle, theta1=0, theta2=90, **marking_args))

        # Draw blue lines.
        bluepatch_args = dict(facecolor="b", edgecolor=None, linewidth=0.0, alpha=0.5, fill=True)
        self.add_patch(mpatches.Rectangle(xy=(-25, -42.5), width=-1, height=85, **bluepatch_args))
        self.add_patch(mpatches.Rectangle(xy=(25, -42.5), width=1, height=85, **bluepatch_args))

        # Draw centre line.
        redline_args = dict(color="r", alpha=0.5)
        redpatch_args = dict(facecolor="r", edgecolor=None, linewidth=0.0, alpha=0.5, fill=True)
        self.add_patch(mpatches.Rectangle(xy=(-0.5, -42.5), width=1, height=85, **redpatch_args))
        self.add_patch(mpatches.Circle(xy=(0, 0), radius=15, fill=False, **redline_args))

        # Draw goal lines.
        gl_extent = 14.5 + np.sqrt(495)  # The half-length of the goaline.
        crease_angle = np.rad2deg(np.arcsin(4.0 / 6))
        self.add_line(Line2D(xdata=(-89, -89), ydata=(-gl_extent, gl_extent), **redline_args))
        self.add_line(Line2D(xdata=(89, 89), ydata=(-gl_extent, gl_extent), **redline_args))
        self.add_line(Line2D(xdata=(-89, -84.5), ydata=(-4, -4), **redline_args))
        self.add_line(Line2D(xdata=(-89, -84.5), ydata=(4, 4), **redline_args))
        self.add_line(Line2D(xdata=(-100, -89), ydata=(-14, -11), **redline_args))
        self.add_line(Line2D(xdata=(-100, -89), ydata=(14, 11), **redline_args))
        self.add_line(Line2D(xdata=(89, 84.5), ydata=(-4, -4), **redline_args))
        self.add_line(Line2D(xdata=(89, 84.5), ydata=(4, 4), **redline_args))
        self.add_line(Line2D(xdata=(100, 89), ydata=(-14, -11), **redline_args))
        self.add_line(Line2D(xdata=(100, 89), ydata=(14, 11), **redline_args))

        self.add_patch(
            mpatches.Arc(
                xy=(-89, 0),
                width=12,
                height=12,
                angle=0,
                theta1=-crease_angle,
                theta2=crease_angle,
                fill=False,
                **redline_args,
            )
        )
        self.add_patch(
            mpatches.Arc(
                xy=(89, 0),
                width=12,
                height=12,
                angle=180,
                theta1=-crease_angle,
                theta2=crease_angle,
                fill=False,
                **redline_args,
            )
        )

        # Goals.
        self.add_patch(mpatches.Rectangle(xy=(-89, -3), width=-2, height=6, color="k", fill=False))
        self.add_patch(mpatches.Rectangle(xy=(89, -3), width=2, height=6, color="k", fill=False))

        # Face-off spots.
        for xy in [(-20, -22), (20, -22), (20, 22), (-20, 22), (0, 0)]:
            self.add_patch(mpatches.Circle(xy, radius=1, **redpatch_args))

        outside_mark_y = np.sqrt(216)
        x_off, x_len = 2, 46.0 / 12
        y_off, y_len = 10.0 / 12, 3

        for xy in [(-69, -22), (69, -22), (69, 22), (-69, 22)]:
            x, y = xy

            # Draw spot and circle.
            self.add_patch(mpatches.Circle(xy, radius=1, **redpatch_args))
            self.add_patch(mpatches.Circle(xy, radius=15, fill=False, **redline_args))

            # Add lines on edge of circle.
            for xdata, ydata in [
                ((x - 3, x - 3), (y - outside_mark_y, y - outside_mark_y - 2)),
                ((x + 3, x + 3), (y - outside_mark_y, y - outside_mark_y - 2)),
                ((x + 3, x + 3), (y + outside_mark_y, y + outside_mark_y + 2)),
                ((x - 3, x - 3), (y + outside_mark_y, y + outside_mark_y + 2)),
            ]:
                self.add_line(Line2D(xdata=xdata, ydata=ydata, **redline_args))

            # Add the elbow markings.
            for xdata, ydata in [
                ((x - x_off, x - x_off), (y - y_off, y - y_off - y_len)),
                ((x + x_off, x + x_off), (y - y_off, y - y_off - y_len)),
                ((x + x_off, x + x_off), (y + y_off, y + y_off + y_len)),
                ((x - x_off, x - x_off), (y + y_off, y + y_off + y_len)),
            ]:
                self.add_line(Line2D(xdata=xdata, ydata=ydata, **redline_args))

            for xdata, ydata in [
                ((x - x_off, x - x_off - x_len), (y - y_off, y - y_off)),
                ((x + x_off, x + x_off + x_len), (y - y_off, y - y_off)),
                ((x + x_off, x + x_off + x_len), (y + y_off, y + y_off)),
                ((x - x_off, x - x_off - x_len), (y + y_off, y + y_off)),
            ]:
                self.add_line(Line2D(xdata=xdata, ydata=ydata, **redline_args))