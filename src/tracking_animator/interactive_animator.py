import ipywidgets as widgets
from matplotlib.pyplot import figure

class InteractiveAnimation(widgets.VBox):
    """
    base class that constructs an interactive plot that allows moving around players/ball
    """
    def __init__(self, basic_plots, df, base_fig, frame_rate=10):
        super().__init__()

        self.animation_container = basic_plots
        self.base_fig = base_fig
        self.frames = df.drop_duplicates('frame_count')
        self.control_container = self.add_to_layout(frame_rate=frame_rate)
    
    def create_animation(self):
        return self.control_container

    def add_to_layout(self, frame_rate):
        """
        add slider elements to to widget container
        """
        no_frames = len(self.frames) - 1
        
        self.play = widgets.Play(interval=1000/frame_rate,
                                value=0,
                                step=1,
                                max=no_frames,
                                description="Press play",
                                disabled=False)
        
        self.slider = widgets.IntSlider(max=no_frames)
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        self.slider.observe(self.__update_data, names='value')
        
        self.phase = widgets.IntText(
            value=0,
            description='Phase: ',
        )
        
        self.time = widgets.FloatText(
            value=0,
            description='Time: ',
        )

        time_step = widgets.HBox([self.phase, self.time])
        slider = widgets.HBox([self.play, self.slider])

        return widgets.VBox([slider, time_step])

    def set_data(self, frame):
        for container in self.animation_container:
            container.set_data(frame)
        self.base_fig.canvas.draw()
    
    def __update_data(self, change):
        """
        update pitch plot
        """
        frame_count = self.frames.iloc[change['new']].frame_count
        
        frame = self.frames[self.frames.frame_count == frame_count]

        self.phase.value = frame.iloc[0].current_phase
        self.time.value = round(frame.iloc[0].timeelapsed, 2)
    
        self.set_data(frame_count)
