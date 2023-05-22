import tkinter as tk
from PIL import ImageTk, Image
import cv2
import time
# from preprocessing.mouth_extraction import extract_mouth
# from preprocessing.img_feature_extraction import load_cnn_vgg_model, extract_instance_features
# from preprocessing.label_tokenization import get_label_tokens
# from models.lstm_model import get_lstm_model
import numpy as np


class App:
    def __init__(self, window):
        self.window = window
        self.video_frames = []
        self.recording = False

        # Create webcam display
        self.webcam_label = tk.Label(window)
        self.webcam_label.pack()

        # Create status label
        self.status_label = tk.Label(window, text='Click "Record" to record an utterance')
        self.status_label.pack()

        # Create record, stop, and preview buttons
        self.record_button = tk.Button(window, text='Record', command=self.start_recording)
        self.record_button.pack(side='left')
        self.stop_button = tk.Button(window, text='Stop', command=self.stop_recording, state='disabled')
        self.stop_button.pack(side='left')
        self.preview_button = tk.Button(window, text='Preview', command=self.preview_video, state='disabled')
        self.preview_button.pack(side='left')

        # Create submit and clear buttons
        self.submit_button = tk.Button(window, text='Submit', command=self.submit_data, state='disabled')
        self.submit_button.pack(side='right')
        self.clear_button = tk.Button(window, text='Clear', command=self.clear_data)
        self.clear_button.pack(side='right')

        # Initialize webcam capture
        self.capture = cv2.VideoCapture(0)
        self.update_webcam()

    def start_recording(self):
        self.video_frames = []
        self.recording = True
        self.status_label.configure(text='Recording... Press "Stop" to end recording')
        self.stop_button.configure(state='normal')
        self.preview_button.configure(state='disabled')
        self.submit_button.configure(state='disabled')

    def stop_recording(self):
        self.recording = False
        self.status_label.configure(text='Video is available for preview')
        self.stop_button.configure(state='disabled')
        self.preview_button.configure(state='normal')
        self.submit_button.configure(state='normal')

    def preview_video(self):
        if self.video_frames:
            cv2.destroyAllWindows()
            for frame in self.video_frames:
                cv2.imshow('Preview', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

    def update_webcam(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip horizontally
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.webcam_label.configure(image=photo)
            self.webcam_label.image = photo
            if self.recording:
                self.video_frames.append(frame)
                self.submit_button.configure(state='disabled')
            elif self.video_frames:
                self.submit_button.configure(state='normal')
        self.window.after(10, self.update_webcam)

    def submit_data(self):
        if self.video_frames:
            self.status_label.configure(text='Processing video for prediction...')
            # custom_function(self.video_frames)  # Replace with your custom function
            # time.sleep(8)
            # self.status_label.configure(text='Prediction for Recording: Have a good time.')
        else:
            print("No video frames captured.")

    def clear_data(self):
        self.video_frames = []
        self.status_label.configure(text='Click "Record" to record an utterance')
        self.stop_button.configure(state='disabled')
        self.preview_button.configure(state='disabled')
        self.submit_button.configure(state='disabled')


def custom_function(video_frames):

    frame_interval = max(1, len(video_frames) // 16)

    # Select 16 frames with equal time intervals
    selected_frames = []
    for i in range(0, len(video_frames), frame_interval):
        selected_frames.append(video_frames[i])

    mouth_frames = [extract_mouth(frame) for frame in selected_frames]

    cnn_model = load_cnn_vgg_model()

    img_feature_stack = extract_instance_features(cnn_model, mouth_frames)
    encoder_input_data = [img_feature_stack]

    decoder_input_data, _ = get_label_tokens([['<start>']])

    encoder_input_data = np.array(encoder_input_data)
    decoder_input_data = np.array(decoder_input_data)

    model = get_lstm_model()
    model.load_weights('best_model.h5')

    prediction = model.predict([encoder_input_data, decoder_input_data])
    print(prediction)


# Create the application window
window = tk.Tk()
window.title("Lip reading demo")
app = App(window)

# Get the screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate the x and y position to center the window
x = (screen_width - 640) // 2
y = (screen_height - 600) // 2

# Set the window position
window.geometry(f"+{x}+{y}")

# Run the application
window.mainloop()
