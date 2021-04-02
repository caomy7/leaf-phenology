import torch
import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torchvision

# path = r"D:\04_Study\02_pytorch_github\10_regression\03_Case\03_face_Age\08_phenocam_regression\dataset\06_dukehw_RGB_Balence\\"
path = r"D:\06_scientific research\11_Image2ROI\Regression_dataset\06_Allsites45_RGB2ROI_14453(OK)"

files = os.listdir(path)
# numTrainData=1000
numTrainData=10000

batch_size =256
# batch_size =31
age =[]
data = []
# print(age)
for file in files:

    m = int(str(file).split("_")[0])
    # print(m)
    age.append(int(str(file).split("_")[0]))

    # print(age)
    phe = path + file
    # image = cv2.imread(phe)
    phenocam = cv2.imread(phe)
    image=phenocam
    # image = phenocam[2:224, 95:212]
    # image = phenocam[120:184, 120:184]
    # img = cv2.imdecode(np.frombuffer(phenocam,np.uint8),cv2.IMREAD_GRAYSCALE)
    image = np.float32(image/255.0)
    # print(image.mode)
    # image = image.permute(64,3,11,11)

    # image = cv2.resize(image,(224,224))
    image = cv2.resize(image,(64,64))
    # print(image.shape)7
    data.append(image)


classes = []
for i in age:

    if (i==0):
        classes.append(0)
    if (i>0) and (i<=8):
        classes.append(1)
    if (i>=9) and (i<=16):
        classes.append(2)
    if (i>=17) and (i<=24):
        classes.append(3)
    if (i>=25) and (i<=32):
        classes.append(4)
    if (i>=33) and (i<=40):
        classes.append(5)
    if (i>=41) and (i<=48):
        classes.append(6)
    if (i>=49) and (i<=56):
        classes.append(7)
    if (i>=57) and (i<=64):
        classes.append(8)
    if (i>=65) and (i<=72):
        classes.append(9)
    if (i>=73) and (i<=80):
        classes.append(10)
    if (i>=81) and (i<=88):
        classes.append(11)
    if (i>=89) and (i<=96):
        classes.append(12)
    if (i>=97) and (i<=104):
        classes.append(13)
    if (i>=105) and (i<=112):
        classes.append(14)
    if (i>=113) and (i<=120):
        classes.append(15)
    if (i>=121) and (i<=128):
        classes.append(16)
    if (i>=129) and (i<=136):
        classes.append(17)
    if (i>=137) and (i<=144):
        classes.append(18)
    if (i>=145) and (i<=152):
        classes.append(19)
    if (i>=153) and (i<=160):
        classes.append(20)
    if (i>=161) and (i<=168):
        classes.append(21)
    if (i>=169) and (i<=176):
        classes.append(22)
    if (i>=177) and (i<=184):
        classes.append(23)
    if (i>=185) and (i<=192):
        classes.append(24)
    if (i>=193) and (i<=200):
        classes.append(25)
    if (i>=201) and (i<=208):
        classes.append(26)
    if (i>=209) and (i<=216):
        classes.append(27)
    if (i>=217) and (i<=224):
        classes.append(28)
    if (i>=225) and (i<=232):
        classes.append(29)
    if (i>=233) and (i<=240):
        classes.append(30)
    if (i>240) and (i<=248):
        classes.append(31)
    if (i>248) and (i<=256):
        classes.append(32)
    if (i>256) and (i<=264):
        classes.append(33)
    if (i>264) and (i<=272):
        classes.append(34)
    if (i>272) and (i<=280):
        classes.append(35)
    if (i>280) and (i<=288):
        classes.append(36)
    if (i>288) and (i<=296):
        classes.append(37)
    if (i>296) and (i<=304):
        classes.append(38)
    if (i>304) and (i<=312):
        classes.append(39)
    if (i>312) and (i<=320):
        classes.append(40)
    # if (i>320) and (i<=328):
    #     classes.append(41)
    # if (i>328) and (i<=336):
    #     classes.append(42)
    # if (i>336) and (i<=344):
    #     classes.append(43)
    # if (i>344) and (i<=352):
    #     classes.append(44)
    # if (i>352) and (i<=360):
    #     classes.append(45)
    # if (i>360) and (i<=368):
    #     classes.append(46)



X = np.squeeze(data)
Y = np.asarray(classes)
# Y = np.asarray(age)
# Y = np.asarray(age)/300.0
# Y = np.asarray(age)/46.0
# print(Y)

print(X.shape,Y.shape)


# Random shuffle data
X, Y = shuffle(X, Y)

print(Y)
# Train-Test-Validation split

train_valid_data = np.array((X[:numTrainData]))

train_valid_labels = np.array((Y[:numTrainData]))

test_data = np.array((X[numTrainData:]))
test_labels = np.array((Y[numTrainData:]))
# print(train_valid_data.shape,test_data.shape)

train_data, valid_data, train_labels, valid_labels = train_test_split(train_valid_data, train_valid_labels, test_size=0.2)


class Dataset(torch.utils.data.Dataset):

    # Create Torch Dataset object.
    def __init__(self , X , Y):
        # X = X.reshape((-1 , 1 , 64 , 64))

        # X = X.reshape((-1 , 3 , 224 ,224))
        # X = X.reshape((-1 , 3 , 222 ,117))（NO）
        X = X.reshape((-1 , 3 , 64 ,64))

        # print(X.shape)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        print(X.shape,Y.shape)

    def __len__(self):
        # print(len(Y))
        return len(self.Y)

    def __getitem__(self , index):
        # print(index)
        # print(self.X.shape)
        # X1 = self.X[2:]
        # print(X1.shape)
        X = self.X[index]
        Y = self.Y[index]

        return {'X': X , 'Y': Y}


class data_enhance(object):

    # def __init__(self, output_size):
    #     assert isinstance(output_size, (int, tuple))

    def __call__(self, sample):
        img= sample['image']
        img = torchvision.transforms.randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))


        img, mask = randomShiftScaleRotate(img,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    # img, mask = randomHorizontalFlip(img, mask)
        img = torchvision.transforms.randomVerticleFlip(img)
        img = torchvision.transforms.randomRotate90(img)
        # mask = np.expand_dims(mask, 2)
        return {'image': img}
# data_enhance(image)

def randomShiftScaleRotate(image,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))


    return image
randomShiftScaleRotate(image)

# def randomHorizontalFlip(image, mask, u=0.5):
def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        # mask = cv2.flip(mask)
randomHorizontalFlip(image)


trainSignData = Dataset(train_data, train_labels)
trainDataLoader = torch.utils.data.DataLoader(trainSignData, shuffle=True, batch_size=batch_size)

testSignData = Dataset(test_data, test_labels)
testDataLoader = torch.utils.data.DataLoader(testSignData, batch_size=4453)

validSignData = Dataset(valid_data, valid_labels)
validDataLoader = torch.utils.data.DataLoader(validSignData, shuffle=True, batch_size=batch_size)
