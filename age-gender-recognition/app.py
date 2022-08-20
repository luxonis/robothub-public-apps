from functools import partial
from pathlib import Path
import re
from typing import List
import time
import numpy as np
import depthai as dai

from robothub_sdk import App, IS_INTERACTIVE, CameraResolution, Config

if IS_INTERACTIVE:
    import cv2

class AgeGender(App):
    def on_initialize(self, unused_devices: List[dai.DeviceInfo]):
        self.msgs = {}
    
    def on_configuration(self, old_configuration: Config):
        pass
    
    def on_setup(self, device):
        camera = device.configure_camera(dai.CameraBoardSocket.RGB, res=CameraResolution.THE_1080_P, \
                preview_size=(1080, 1080))
        stereo = device.create_stereo_depth()
        (_, nn_det_out, nn_det_passthrough) = device.create_nn(source=device.streams.color_preview, blob_path=Path("./det_model.blob"), \
            config_path=Path("./det_model.json"), input_size=(300, 300), depth=stereo)

        (manip, manip_stream) = device.create_image_manipulator()
        manip.initialConfig.setResize(62, 62)
        manip.inputConfig.setWaitForMessage(True)
        self.script = device.create_script(script_path=Path("./script.py"),
            inputs={
                'preview': device.streams.color_preview,
                'passthrough' : nn_det_passthrough,
                'face_det_in' : nn_det_out
            },
            outputs={
                'manip_img': manip.inputImage,
                'manip_cfg': manip.inputConfig
            })
        
        (_, nn_age_out, nn_age_passthrough) = device.create_nn(source=manip_stream, blob_path=Path("./age_model.blob"), \
            config_path=Path("./age_model.json"), input_size=(62, 62))
        
        if IS_INTERACTIVE:
            device.streams.color_preview.consume(partial(self.add_msg, 'color'))
            nn_det_out.consume(partial(self.add_msg, 'detection'))
            nn_age_out.consume(partial(self.add_msg, 'recognition'))
  
    def on_update(self):
        if IS_INTERACTIVE:
            for device in self.devices:
                msgs = self.get_msgs()
                if msgs is not None:
                    frame = msgs["color"].getCvFrame()
                    detections = msgs["detection"].detections
                    recognitions = msgs["recognition"]

                    for i, detection in enumerate(detections):
                        bbox = self.frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                        # Decoding of recognition results
                        rec = recognitions[i]
                        age = int(float(np.squeeze(np.array(rec.getLayerFp16('age_conv3')))) * 100)
                        gender = np.squeeze(np.array(rec.getLayerFp16('prob')))
                        gender_str = "female" if gender[0] > gender[1] else "male"

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                        y = (bbox[1] + bbox[3]) // 2
                        cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                        cv2.putText(frame, str(age), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                        cv2.putText(frame, gender_str, (bbox[0], y + 30), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 8)
                        cv2.putText(frame, gender_str, (bbox[0], y + 30), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 2)
                        coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                        cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                        cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

                    cv2.imshow("Camera", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                self.stop()

    def frame_norm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def add_msg(self, name, msg : dai.NNData):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {}
        if 'recognition' not in self.msgs[seq]:
            self.msgs[seq]['recognition'] = []
        if name == 'recognition':
            self.msgs[seq]['recognition'].append(msg)
        elif name == 'detection':
            self.msgs[seq][name] = msg
            self.msgs[seq]["len"] = len(msg.detections)
        elif name == 'color':
            self.msgs[seq][name] = msg

    def get_msgs(self):
        seq_remove = []
        for seq, msgs in self.msgs.items():
            seq_remove.append(seq) 
            if "color" in msgs and "len" in msgs:
                if msgs["len"] == len(msgs["recognition"]):
                    for rm in seq_remove:
                        del self.msgs[rm]
                    return msgs
        return None

app = AgeGender()
app.run()
