import os
import inference

model = inference.get_roboflow_model("cargotrack/3")
results = model.infer(image="/home/gkap/repos/hacktech/frame_10.jpg")
print(results)