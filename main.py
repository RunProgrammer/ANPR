
import cv2
import cv2
import argparse
from utils import *
from torch_utils import *
from darknet2pytorch import Darknet
#from tool_utils import *
#from VideoCapture import VideoCapture


def str2int(source):

    try:
        return int(source)

    except ValueError:
        return source


def run_inference(cfgfile, weightfile, namesfile, source, output, conf, nms, save_net):
    model = Darknet(cfgfile)
    model.print_network()

    if save_net != " ":
        model.save_weights(outfile=save_net)

    try:
        model.load_weights(weightfile)
    except Exception:
        print("Could not load Weights")

    cuda = torch.cuda.is_available()

    if cuda:
        print("CUDA found running on GPU")
        model.cuda()

    print("CUDA not found running on CPU")

    model.eval()
    with torch.no_grad():
        source = str2int(source)
        cap = cv2.VideoCapture(source)
        harcascade = "haarcascade_russian_plate_number.xml"

        #cap = cv2.VideoCapture(0)

        cap.set(3, 640) # width
        cap.set(4, 480) #height

        min_area = 500
        count = 0

        while True:
            success, img = cap.read()

            plate_cascade = cv2.CascadeClassifier(harcascade)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
            print('Number of detected license plates:', len(plates))
            for (x,y,w,h) in plates:

   
   # draw bounding rectangle around the license number plate
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                gray_plates = img_gray[y:y+h, x:x+w]
                color_plates = img[y:y+h, x:x+w]
                
                # save number plate detected
                cv2.imwrite('Numberplate.jpg', gray_plates)
                cv2.imshow('Number Plate', gray_plates)
                cv2.imshow('Number Plate Image', img)
                cv2.waitKey(0)
                    
        width = int(cap.get(3))
        height = int(cap.get(4))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        """ Load the labels from the .names file """
        class_names = load_class_names(namesfile)

        if output != " ":
            out = cv2.VideoWriter(
                output,
                cv2.VideoWriter_fourcc("X", "2", "6", "4"),
                frame_rate,
                (width, height),
            )

        while True:
            ret, img = cap.read()

            if not ret:
                exit(0)

            try:

           
                img_resized = cv2.resize(img, (model.width, model.height))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

                start = time.time()
                boxes = do_detect(model, img_rgb, conf, nms, cuda)

                print("predicted in %f seconds." % (time.time() - start))

               
                antd_img = plot_boxes_cv2(
                    img, boxes[0], savename=None, class_names=class_names
                )

               
                fps = int(1 / (time.time() - start))
                print(f"FPS: {fps}")

            except Exception:
                pass

            if output != " ":
                out.write(antd_img)

            cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)

            """ Show the frame """
            cv2.imshow("Inference", antd_img)

            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break

       
        cap.release()
        if output != " ":
            out.release()
        cv2.destroyAllWindows()


def arguments():

  
    parser = argparse.ArgumentParser("Arguments for running inference.")

    parser.add_argument(
        "-cfgfile",
        type=str,
        default="yolov4.cfg",
        help="Path to the configuration file",
        dest="cfgfile",
    )

    parser.add_argument(
        "-weightfile",
        type=str,
        default="./weights/yolov4.weights",
        help="Path to the weights file",
        dest="weightfile",
    )

    parser.add_argument(
        "-namesfile",
        type=str,
        default="classes.names",
        help="Path to the classes name file",
        dest="namesfile",
    )

    parser.add_argument(
        "-source",
        type=str,
        default=0,
        help="Source for webcam default 0 for the built in webcam",
        dest="source",
    )

    parser.add_argument(
        "-output",
        type=str,
        default=" ",
        help="Name of the output result video",
        dest="output",
    )

    parser.add_argument(
        "-conf",
        type=float,
        default=0.4,
        help="Confidence threshold for inference",
        dest="conf_thresh",
    )

    parser.add_argument(
        "-nms",
        type=float,
        default=0.6,
        help="Non maximum supression threshold",
        dest="nms_thresh",
    )

    parser.add_argument(
        "-save_weight",
        type=str,
        default=" ",
        help="Save weight to pytorch format output weight file name",
        dest="save_net",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = arguments()

    """ Start the inference function """
    run_inference (
        args.cfgfile,
        args.weightfile,
        args.namesfile,
        args.source,
        args.output,
        args.conf_thresh,
        args.nms_thresh,
        args.save_net,
    
    )
