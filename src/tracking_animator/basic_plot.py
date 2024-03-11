import numpy as np
from .visualiser import *
import matplotlib.patches as patches
import ipywidgets as widgets

polygon_columns = ['p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y']
DEFAULT_COLORS = {1: 'navy', 2: 'orange', 4: 'red'}

class BasicPlot(widgets.VBox):
    """
    base class that constructs a plot of a tracking frame
    """
    def __init__(
        self, 
        ax, 
        df, 
        title, 
        color_map=None, 
        is_draw_events=True, 
        is_draw_polygon=True, 
        is_draw_tail=False,
        is_draw_speed=False    
    ):
        """
        init widget object
        """
        super().__init__()

        self.ax = ax
        self.ax.set_title(title)
        self.title = title
        self.frame_artists = []
        self.parents = []
        self.is_draw_polygon = is_draw_polygon
        self.is_draw_events = is_draw_events
        self.is_draw_tail = is_draw_tail
        self.is_draw_speed = is_draw_speed
        self.df = df
        self.full_df = df
        self.extra_artists = []

        if color_map is None:
            self.color_map = DEFAULT_COLORS
        else:
            self.color_map = color_map
        self.current_frame = -1

        self.agent_information = {
            'full': self.init_locations('full'),
            'empty': self.init_locations('empty'),
            'event': self.init_event(),
            'trapezoid': self.init_trapezoid()
        }

    def init_locations(self, src):
        information = {}

        players = self.full_df.drop_duplicates('player_id')
        for _, player in players.iterrows():
            color = self.color_map[int(player.team_id)]
            is_ball = int(player.team_id) != 4
            
            args = {
                "xy": (0, 0),
                "label": int(player.jersey_no) if is_ball else "",
                "font_size": 10,
                "zorder": 99 if player.is_ball else 90,
                "radius": 2 if is_ball else 1
            }
            
            args['edgecolor'] = color
            if src == 'full':
                args['facecolor'] = color
            else:
                args['fill'] = False
        
            if self.is_draw_tail:
                args['max_trace_length'] = 10

            if self.is_draw_tail:
                cls = PlayerArtistWithTrace
            elif self.is_draw_speed:
                cls = PlayerArtistWithVelocity
            else:
                cls = PlayerArtist

            artist = cls(**args)
            artist.set_visible(False)
            self.ax.add_artist(artist)
            information[player.player_id] = artist
        return information
    
    def init_event(self):
        event_artist = PlayerArtist(
            (0, 0), 
            radius=1, 
            color='white',
            visible=False
        )

        self.ax.add_artist(event_artist)
        text_artist = self.ax.text(
            0, 0,
            '',
            fontsize=10,
            visible=False
        )
        self.ax.add_artist(text_artist)

        return {
            'circle': event_artist,
            'text': text_artist
        }
    
    def init_trapezoid(self):
        return self.ax.add_patch(
            patches.Polygon(
            xy=[(0,0),(1,1),(1,0)], 
            fill=True, 
            facecolor=(1, 1, 1, 0.7), 
            zorder=-1,
            edgecolor=(0.8, 0.8, 0.8))
        )
    
    def draw_location(self, player, src, draw_tails):
        player_id = player.player_id

        if not np.isnan(player[f"{src}_x"]):                
            artist = self.agent_information[src][player.player_id] 
            if not draw_tails and self.is_draw_tail:
                artist.trace.set_segments([])

            artist.set_center((player[f"{src}_x"], player[f"{src}_y"]))
            artist.set_visible(True)

            if self.is_draw_speed:
                artist.set_velocity((player[f"{src}_speed_x"], player[f"{src}_speed_y"]))

        else:
            if player_id in self.agent_information[src]:
                artist = self.agent_information[src][player.player_id] 
                artist.set_visible(False)

    def draw_polygon(self, polygon):
        x_features = [x for x in polygon_columns if x.endswith('_x')]
        y_features = [x for x in polygon_columns if x.endswith('_y')]
        
        x = polygon[x_features]
        y = polygon[y_features]

        if not x.isna().any():
            self.agent_information['trapezoid'].set(xy=list(zip(x, y)))
            self.agent_information['trapezoid'].set_visible(True)
        else:
            self.agent_information['trapezoid'].set_visible(False)

    def draw_last_event(self, frame):
        frame_id = frame.iloc[0].frame_count
        phase = frame.iloc[0].current_phase
        
        events = self.df[~self.df.event_description.isna()]
        previous_events = events[
            (events.current_phase == phase) & 
            (events.frame_count <= frame_id) & 
            (events.event_description != 'deleted_event')
        ]

        if not previous_events.empty:
            recent_event = previous_events.iloc[-1]
            color = self.color_map[recent_event.team_id]

            self.agent_information['event']['circle'].set_center((recent_event.x, recent_event.y))
            self.agent_information['event']['circle'].set(color=color)
            self.agent_information['event']['text'].set(x=recent_event.x+1, y=recent_event.y+1, color='black')
            self.agent_information['event']['text'].set_text(recent_event.event_description)

            self.agent_information['event']['circle'].set_visible(True)
            self.agent_information['event']['text'].set_visible(True)

        else:
            self.agent_information['event']['circle'].set_visible(False)
            self.agent_information['event']['text'].set_visible(False)


    def set_data(self, frame_id):
        """
        set data of scatter plots
        input:
          selected_frame: tuple consisting of (current_phase, timeelapsed)  
        """

        frame = self.df[self.df.frame_count == frame_id]
        draw_tails = (frame_id == self.current_frame + 1)
        self.current_frame = frame_id

        if frame.empty:
            return
        
        if self.is_draw_polygon:
            self.draw_polygon(frame.iloc[0])
        
        if self.is_draw_events:
            self.draw_last_event(frame)
        
        for _, row in frame.iterrows():
            self.draw_location(row, 'full', draw_tails)
            self.draw_location(row, 'empty', draw_tails)
        
        visible_in_frame = set(frame.player_id.unique())
        all_players = set(self.agent_information['full'].keys())
        non_visible = all_players - visible_in_frame

        for player_id in non_visible:
            self.agent_information['full'][player_id].set_visible(False)
            self.agent_information['empty'][player_id].set_visible(False)
 
        for parent in self.parents:
            if hasattr(parent, 'set_data'):
                parent.set_data(frame, self)        
            