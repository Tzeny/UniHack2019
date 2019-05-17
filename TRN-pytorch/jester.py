import time

import cv2
import torch
from PIL import Image
from torch.functional import F

import transforms
import torchvision
from models import TSN
filepath = "pretrain/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar"
m = torch.load(filepath)
categories_file = 'pretrain/{}_categories.txt'.format('jester')
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)
arch = "InceptionV3"
test_segments = 8
modality = "RGB"
consensus_type = "TRNmultiscale"
img_feature_dim = 256
torch.backends.cudnn.benchmark = True
net = TSN(num_class,
          test_segments,
          modality,
          base_model=arch,
          consensus_type=consensus_type,
          img_feature_dim=img_feature_dim,
          print_spec=False)
def get_predictions_for_8_frames(net, frames):
    a_t = time.time()

    args_arch = "BNInception"
    transform = torchvision.transforms.Compose([
        transforms.GroupOverSample(net.input_size, net.scale_size),
        transforms.Stack(roll=(args_arch in ['BNInception', 'InceptionV3'])),
        transforms.ToTorchFormatTensor(div=(args_arch not in ['BNInception', 'InceptionV3'])),
        transforms.GroupNormalize(net.input_mean, net.input_std),
    ])

    data = transform(frames)
    input = data.view(-1, 3, data.size(1),
                      data.size(2)).unsqueeze(0)
    with torch.no_grad():
        logits = net(input)
        torch.onnx.export(net, input, "plm.onnx", verbose=True, input_names=["input"], output_names=["output"])
        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)

    b_t = time.time()

    print(f'Elapsed: {b_t - a_t}')

    # Output the prediction.
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
print("Initializing webcam.")
cap = cv2.VideoCapture(0)
current_batch = []
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.resize(frame, (256, 256))
    cv2.imshow('frame', frame)

    current_batch.append(Image.fromarray(frame))

    if len(current_batch) == 8:
        get_predictions_for_8_frames(net, current_batch)
        current_batch = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


