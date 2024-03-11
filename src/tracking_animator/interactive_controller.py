import ipywidgets as widgets
import qgrid
import pandas as pd
import numpy as np

event_columns = [
    'current_phase',
    'timeelapsed',
    'team_id',
    'jersey_no',
    'x',
    'y',
    'event_description',
    'frame_count',
    'player_id'
]        

class InteractiveController(widgets.VBox):
    def __init__(self, interactive, df):
        super().__init__()
        
        self.animation_container = interactive
        self.df = df
        self.plot_names = [t.title for t in self.animation_container.animation_container]
        self.df_events = self.df[self.df.typeId != 0][
            event_columns
        ].reset_index()

        for i in self.animation_container.animation_container:
            i.parents.append(self)
    
    def create_animation(self):
        self.segment_num = widgets.Dropdown(
            description='Segment #',
            options=pd.unique(self.df['current_phase']),
            value=pd.unique(self.df['current_phase'])[0]
        )
        self.segment_num.observe(self.__shot_num_changed, names='value')
        self.stopped = widgets.Text(
            value='',
            description='Play stopped: '
        )
        
        qgrid.set_grid_option('maxVisibleRows', 10)

        self.__event_container = qgrid.show_grid(self.df_events)
        self.__event_container.layout = widgets.Layout(width='920px')
        self.__event_container.observe(self.__on_row_selected, names=['_selected_rows'])

        return widgets.VBox([self.animation_container.control_container, 
                             self.stopped, 
                             self.segment_num,
                             self.__event_container])

    def __on_row_selected(self, change):
        """
        callback for row selection: update selected points in scatter plot
        """
        # get selcted event
        filtered_df = self.__event_container.get_changed_df()
        event = filtered_df.iloc[change.new].iloc[0]

        index = np.where(self.animation_container.frames.frame_count == event['frame_count'])[0][0]
        self.animation_container.slider.value = index

    def __shot_num_changed(self, value):
        shot = self.df[self.df['current_phase'] == value['new']]
        for basic_plot in self.animation_container.animation_container:
            basic_plot.df = basic_plot.full_df[basic_plot.full_df['current_phase'] == value['new']]
            
        frames = shot.drop_duplicates('frame_count')
        
        self.animation_container.play.max = len(frames) - 1
        self.animation_container.slider.max = len(frames) - 1

        self.animation_container.frames = frames
        self.animation_container.df = shot
        
        view = self.df_events[self.df_events['current_phase'] == int(value.new)]
        self.__event_container.df = view
        
    def set_data(self, frame, basic_plot):
        self.stopped.value = str(frame.stoppage.iloc[0])