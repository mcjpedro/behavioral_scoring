import os
import json
import cv2
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use("TkAgg")  # Ensure consistent backend

class BehaviorAnnotation:
    def __init__(self, config = None):
        if config is None:
            launcher = AnnotationLauncher()
            config = launcher.run()
            
            if config is None:
                print("No configuration provided. Exiting.")
                return
            
        if 'video_path' not in config:
            raise ValueError("Invalid configuration provided.")
        
        self.video_path = config['video_path']
        self.chunk_size = config.get('chunk_size', 500)
        self.labels = config.get('labels', None)
        self.behavior_names = config.get('behavior_names', ["non_assigned"])
        self.behavior_tags = config.get('behavior_tags', list(range(len(self.behavior_names))))
        self.current_frame = config.get('current_frame', 0)
        self.object_rois = config.get('object_rois', {})
        self.location_path = config.get('location_path', None)
        
        if self.location_path is not None:
            self.location = self._load_location(self.location_path)
        else:
            self.location = config.get('location', None)       

        self.behavioral_dict = {tag: name for tag, name in zip(self.behavior_tags, self.behavior_names)}

        # --- 3. ENGINE INITIALIZATION ---
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("Could not open AVI video.")

        self.folder_path = os.path.dirname(self.video_path)
        self.name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_shape = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        if self.labels is None:
            self.labels = np.zeros(self.n_frames, dtype=int)
            self.label_names = ["non_assigned"] * self.n_frames
        elif len(self.labels) != self.n_frames:
            raise ValueError("Length of labels array must match the number of video frames.")

        self.ms_per_frame = 30 
        self.is_playing = False
        self.label_changed = False
        self.n_behaviors = len(self.behavior_names)
        self.video_chunk = None       
        self.chunk_start = -1         
        
        # Colormap Setup (White for index 0, Tab10 for others)
        colors = ['#ffffff'] + [mcolors.to_hex(plt.cm.tab10(i)) for i in range(self.n_behaviors - 1)]    
        self.cmap = mcolors.ListedColormap(colors)
        
        # Fast Rendering Handles
        self.im_h = None
        self.txt_h = None
        self.inside_roi = np.zeros(self.n_frames, dtype=bool)
        
        # Rebuild proximity logic if ROIs and Location exist
        if self.object_rois and self.location is not None:
            self._rebuild_proximity_logic()

        self._setup_ui()

    def _load_location(self, path):
        if isinstance(path, str) and os.path.exists(path):
            if path.endswith('.npy'):
                return np.load(path)
            elif path.endswith('.csv'):
                # read from exTracted CSV format (expects columns 'X' and 'Y')
                df = pd.read_csv(path)
                if 'X' in df.columns and 'Y' in df.columns:
                    return df[['X', 'Y']].values
                else:                  
                    raise ValueError("CSV location file must contain 'X' and 'Y' columns.")
        elif isinstance(path, np.ndarray):
            return path
        return None

    def _rebuild_proximity_logic(self):
        self.inside_roi = np.zeros(self.n_frames, dtype=bool)
        for mask in self.object_rois.values():
            dist_map = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
            # Clip tracking data to image boundaries
            x_coords = np.clip(self.location[:, 0].astype(int), 0, self.img_shape[1] - 1)
            y_coords = np.clip(self.location[:, 1].astype(int), 0, self.img_shape[0] - 1)
            is_close = dist_map[y_coords, x_coords] <= 100
            self.inside_roi[:len(is_close)] = np.logical_or(self.inside_roi[:len(is_close)], is_close)

    def _load_video_batch(self, start_idx):
        end_idx = min(start_idx + self.chunk_size, self.n_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        batch = []
        for _ in range(end_idx - start_idx):
            ret, frame = self.cap.read()
            if not ret: break
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.video_chunk = np.array(batch)
        self.chunk_start = start_idx

    def _get_frame(self, idx):
        if self.video_chunk is None or not (self.chunk_start <= idx < self.chunk_start + len(self.video_chunk)):
            self._load_video_batch(idx)
        return self.video_chunk[idx - self.chunk_start]

    def _setup_ui(self):
        self.fig = plt.figure(figsize=(7, 8))
        gs = self.fig.add_gridspec(6, 1)

        self.ax_video = self.fig.add_subplot(gs[0:5, :])
        self.ax_playback = self.fig.add_subplot(gs[5, :]) 

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('close_event', self._handle_close_request)
        
        self.timer = self.fig.canvas.new_timer(interval=self.ms_per_frame)
        self.timer.add_callback(self._play_step)

        self.update_all()
        plt.show(block=True)

    def update_all(self):
        for ax in [self.ax_video, self.ax_playback]:
            ax.cla()

        self.fig.suptitle(
            f"Behavior Annotation Tool\n(P) Play/Pause | (Left/Right) Navigate | (+/-) Speed \n (E) Save | (O) Draw ROI | (I) Delete ROIs\nBehaviors: {', '.join([f'{i}:{name}' for i, name in enumerate(self.behavior_names)])}",
            fontsize=10)

        self.ax_playback.set_xlim(0, self.n_frames)
        self.im_pl = self.ax_playback.imshow(self.labels.reshape(1, -1), aspect='auto', cmap=self.cmap, vmin=0, vmax=self.n_behaviors - 1, extent=[0, self.n_frames, -0.5, 0.5], animated=True)
        self.ax_playback.set_yticks([])
        for spine in self.ax_playback.spines.values():
            spine.set_visible(False)
        
        self.line_h = self.ax_playback.axvline(self.current_frame, color='red', lw=0.5, animated=True)

        frame = self._get_frame(self.current_frame)
        self.im_h = self.ax_video.imshow(frame, cmap='gray', interpolation='nearest', animated=True)
        self.ax_video.axis('off')

        self.roi_artists = [] 
        if len(self.object_rois) > 0:
            for roi_mask in self.object_rois.values():
                if isinstance(roi_mask, np.ndarray) and roi_mask.ndim == 2:
                    cnt = self.ax_video.contour(roi_mask, levels=[0.5], colors='maroon', linewidths=1)
                    # Note: Contour artists don't always support set_animated directly like others
                    self.roi_artists.append(cnt)

        self.txt_h = self.ax_video.text(0.1, 0.9, f"F: {self.current_frame}", transform=self.ax_video.transAxes, color='red', animated=True)

        self.fig.canvas.draw()
        self.background_video = self.fig.canvas.copy_from_bbox(self.ax_video.bbox)
        
        self.line_h.set_visible(False)
        self.fig.canvas.draw_idle()
        self.background_playback = self.fig.canvas.copy_from_bbox(self.ax_playback.bbox)
        self.line_h.set_visible(True)

    def update_fast(self):
        if self.im_h is None or self.line_h is None:
            return

        self.fig.canvas.restore_region(self.background_video)
        self.fig.canvas.restore_region(self.background_playback)

        self.im_h.set_data(self._get_frame(self.current_frame))
        if self.label_changed:
            self.im_pl.set_data(self.labels.reshape(1, -1))
            
        self.line_h.set_xdata([self.current_frame, self.current_frame])
        lbl_idx = self.labels[self.current_frame]
        label_text = self.behavior_names[lbl_idx]
        self.txt_h.set_text(f"F: {self.current_frame} | Label: {label_text} | {self.ms_per_frame}ms")

        self.ax_video.draw_artist(self.im_h)
        for cnt in self.roi_artists:
            # Re-drawing contours for blitting is complex; usually done via full draw if ROI changed
            self.ax_video.draw_artist(cnt)
        self.ax_video.draw_artist(self.txt_h)
        
        if self.label_changed:
            self.ax_playback.draw_artist(self.im_pl)
            self.background_playback = self.fig.canvas.copy_from_bbox(self.ax_playback.bbox)
            self.label_changed = False
            
        self.ax_playback.draw_artist(self.line_h)

        self.fig.canvas.blit(self.ax_video.bbox)
        self.fig.canvas.blit(self.ax_playback.bbox)
        self.fig.canvas.flush_events()

    def on_click(self, event):
        if event.inaxes == self.ax_playback and event.xdata is not None:
            self.current_frame = int(np.clip(event.xdata, 0, self.n_frames - 1))
            self.update_fast()

    def _play_step(self):
        if self.current_frame < self.n_frames - 1:
            self.current_frame += 1
            self.update_fast()
        else:
            self.timer.stop()
            self.is_playing = False

    def on_key(self, event):
        if event.key in [str(i) for i in self.behavioral_dict]:
            self.labels[self.current_frame] = int(event.key)
            if self.is_playing:
                self.timer.stop()
                self.is_playing = False
            if self.current_frame < self.n_frames - 1:
                self.current_frame += 1
            self.label_changed = True
            self.update_fast()

        elif event.key == 'p':
            if self.is_playing: self.timer.stop()
            else: self.timer.start()
            self.is_playing = not self.is_playing

        elif event.key == 'right':
            self.current_frame = min(self.n_frames - 1, self.current_frame + 1)
            self.update_fast()

        elif event.key == 'left':
            self.current_frame = max(0, self.current_frame - 1)
            self.update_fast()

        elif event.key == '+':
            self.ms_per_frame = max(1, self.ms_per_frame - 5)
            self.timer.interval = self.ms_per_frame
            self.update_fast()

        elif event.key == '-':
            self.ms_per_frame += 5
            self.timer.interval = self.ms_per_frame
            self.update_fast()

        elif event.key == 'e':
            self.save_annotation()

        elif event.key == 'o':
            self.insert_object_roi()

        elif event.key == 'i':
            self.object_rois = {}
            self.inside_roi = np.zeros(self.n_frames, dtype=bool)
            self.update_all()

        elif event.key == 'up':
            next_frames = np.where(self.inside_roi)[0]
            next_frames = next_frames[next_frames > self.current_frame]
            if len(next_frames) > 0:
                self.current_frame = next_frames[0]
                self.update_fast()

        elif event.key == 'down':
            prev_frames = np.where(self.inside_roi)[0]
            prev_frames = prev_frames[prev_frames < self.current_frame]
            if len(prev_frames) > 0:
                self.current_frame = prev_frames[-1]
                self.update_fast()

        elif event.key == 'q':
            plt.close(self.fig)

    def _handle_close_request(self, event=None):
        root = tk.Tk()
        root.withdraw() 
        response = messagebox.askyesno("Quit", "Save session before quitting?")
        if response is True:
            self.save_annotation()
            self._final_cleanup()
        else:
            self._final_cleanup()
        root.destroy()

    def _final_cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        plt.close(self.fig)

    def insert_object_roi(self):
        new_behavior_idx = len(self.behavioral_dict)
        if new_behavior_idx > 9:
            print("Maximum limit (10 behaviors) reached.")
            return
        
        if self.location is None:
            print("No location data available to calculate proximity.")
            return

        frame_image = self._get_frame(self.current_frame).copy()
        pts = []

        def draw_roi(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and len(pts) > 0:
                pts.pop()

        win_name = "Draw Polygon - ENTER to finish, ESC to cancel"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, draw_roi)

        while True:
            img_copy = cv2.cvtColor(frame_image, cv2.COLOR_GRAY2BGR)
            for p in pts:
                cv2.circle(img_copy, p, 3, (0, 0, 255), -1)
            if len(pts) > 1:
                cv2.polylines(img_copy, [np.array(pts)], False, (0, 255, 0), 2)
            cv2.imshow(win_name, img_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 13: break
            if key == 27: pts = []; break

        cv2.destroyWindow(win_name)

        if len(pts) > 2:
            roi_mask = np.zeros(self.img_shape, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(pts)], 1)
            
            obj_name = f"object_{len(self.object_rois)}"
            self.object_rois[obj_name] = roi_mask
            
            self.behavioral_dict[new_behavior_idx] = obj_name
            self.behavior_names.append(obj_name)
            self.n_behaviors += 1
            
            self._update_colormap()
            self._rebuild_proximity_logic()
            self.update_all()

    def _update_colormap(self):
        """
        Updates the colormap based on the current number of behaviors.
        Index 0 is always white (non_assigned).
        Indices 1-9 use the Matplotlib 'tab10' qualitative color palette.
        """
        # Create a list of hex colors
        # Index 0 = White
        # cm.tab10(i) provides a distinct color for each behavior index
        colors = ['#ffffff'] + [mcolors.to_hex(plt.cm.tab10(i)) for i in range(self.n_behaviors - 1)]
        
        # Create the ListedColormap object
        self.cmap = mcolors.ListedColormap(colors)
        
        # If the playback image already exists, update its colormap and limits
        if hasattr(self, 'im_pl'):
            self.im_pl.set_cmap(self.cmap)
            self.im_pl.set_clim(0, self.n_behaviors - 1)

    def save_annotation(self):
        """Saves the complete session to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.folder_path, f"{self.name}_session_{timestamp}.json")
        
        serialized_rois = {name: np.array(mask).astype(int).tolist() for name, mask in self.object_rois.items()}
        
        data = {
            'video_path': self.video_path,
            'chunk_size': self.chunk_size,
            'labels': self.labels.tolist(),
            'behavior_names': self.behavior_names,
            'behavior_tags': [l for l in self.behavioral_dict],
            'last_frame': self.current_frame,
            'location': self.location.tolist() if self.location is not None else None,
            'object_rois': serialized_rois
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f)
        print(f"Session saved successfully: {save_path}")

class AnnotationLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Behavior Annotator Launcher")
        self.root.geometry("400x480")
        self.root.attributes('-topmost', True)
        
        self.final_config = None
        self.temp_location_path = None
        self.mode_var = tk.StringVar(value="new")
        
        self._build_ui()
        
    def _build_ui(self):
        # --- UI Helper Functions ---
        def toggle_inputs():
            """Blocks or enables inputs based on the selected mode."""
            state = "disabled" if self.mode_var.get() == "load" else "normal"
            color = "gray" if self.mode_var.get() == "load" else "black"
            
            self.behav_entry.config(state=state)
            self.chunk_entry.config(state=state)
            self.loc_button.config(state=state)
            self.behav_label.config(fg=color)
            self.chunk_label.config(fg=color)

        tk.Label(self.root, text="Select Start Mode:", font=('Arial', 12, 'bold')).pack(pady=15)
        
        tk.Radiobutton(self.root, text="New Project (AVI Video)", variable=self.mode_var, value="new", command=toggle_inputs).pack(anchor='w', padx=50)
        tk.Radiobutton(self.root, text="Load Session (JSON File)", variable=self.mode_var, value="load", command=toggle_inputs).pack(anchor='w', padx=50)

        self.new_frame = tk.LabelFrame(self.root, text="Configuration (New Project Only)", padx=10, pady=10)
        self.new_frame.pack(pady=20, fill="x", padx=20)

        self.behav_label = tk.Label(self.new_frame, text="Behaviors (comma separated):")
        self.behav_label.pack()
        self.behav_entry = tk.Entry(self.new_frame, width=35)
        self.behav_entry.insert(0, "sniffing, grooming, walking")
        self.behav_entry.pack(pady=5)

        self.chunk_label = tk.Label(self.new_frame, text="Chunk Size (frames):")
        self.chunk_label.pack()
        self.chunk_entry = tk.Entry(self.new_frame, width=10)
        self.chunk_entry.insert(0, "500")
        self.chunk_entry.pack(pady=5)

        self.loc_button = tk.Button(self.new_frame, text="Link Location File (Optional)", command=self._select_loc)
        self.loc_button.pack(pady=10)

        # IMPORTANT: This button calls the logic-heavy _on_launch
        tk.Button(self.root, text="OPEN ANNOTATOR", command=self._on_launch, bg="#3498db", fg="white", 
                  font=('Arial', 10, 'bold'), height=2).pack(pady=20)
        
        toggle_inputs()

    def _select_loc(self):
        path = filedialog.askopenfilename(title="Select Location Data", filetypes=[("Data", "*.csv *.npy")])
        if path:
            self.temp_location_path = path
            self.loc_button.config(bg="#166637", fg="white", text="Location Linked ✓")

    def _on_launch(self):
        """Processes the logic for New vs Load before closing."""
        try:
            if self.mode_var.get() == 'load':
                json_path = filedialog.askopenfilename(title="Select Session JSON File", filetypes=[("JSON", "*.json")])
                if not json_path: return
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                self.final_config = {
                    'video_path': data['video_path'],
                    'chunk_size': data['chunk_size'],
                    'labels': np.array(data['labels']),
                    'behavior_names': data['behavior_names'],
                    'behavior_tags': data['behavior_tags'],
                    'current_frame': data['last_frame'],
                    'location': np.array(data['location']) if data.get('location') is not None else None,
                    'object_rois': {str(key): np.array(mask, dtype=np.uint8) for key, mask in data.get('object_rois', {}).items()}
                }
            else:
                video_path = filedialog.askopenfilename(title="Select AVI Video", filetypes=[("Video", "*.avi")])
                if not video_path: return
                
                raw_names = [s.strip() for s in self.behav_entry.get().split(",")]
                behavior_names = ["non_assigned"] + [n for n in raw_names if n != "non_assigned"]
                
                self.final_config = {
                    'video_path': video_path,
                    'chunk_size': int(self.chunk_entry.get()),
                    'behavior_names': behavior_names,
                    'behavior_tags': list(range(len(behavior_names))),
                    'current_frame': 0,
                    'location_path': self.temp_location_path,
                    'location': None,
                    'object_rois': {},
                }
            
            # This is the only place destroy should be called
            self.root.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize session: {e}")

    def run(self):
        self.root.mainloop()
        return self.final_config
    
if __name__ == "__main__":
    app = BehaviorAnnotation()