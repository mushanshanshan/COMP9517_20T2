import numpy as np
import cv2
import os
import random


def data_normalization(img, method):
    if method in ['HE', 'he', 'hist']:
        img = cv2.equalizeHist(img).astype(np.float)
        return img / 255 - 0.5
    elif method in ['CS', 'cs', 'contrast']:
        img = contrast_stretch(img, None).astype(np.float)
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
        bg = np.argmax(hist)
        img[img <= bg] = 0
        img = cv2.equalizeHist(img)
        return img / 255 - 0.5
    elif method in ['MS', 'ms', 'median']:
        median = np.median(img)
        img = img.astype(np.float) - median
        median *= 2
        img[img < 0] /= median
        img[img > 0] /= 510 - median
        return img


def img_rotate_scale(img, angle, scale=1):
    h, w = img.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def data_augmentation(imgs):
    # random.seed(0)
    angle = random.randint(-180, 180)
    scale = random.uniform(0.6, 1.4)
    flipCode = round(random.random())
    result = []
    for img in imgs:
        img = img_rotate_scale(img, angle, scale)
        result.append(cv2.flip(img, flipCode))
    return result


def contrast_stretch(img, dtype=np.uint8):
    a, b = np.min(img), np.max(img)
    hist, bins = np.histogram(img.ravel(), b - a + 1, [a, b + 1])
    c, d = (hist != 0).argmax(), b - a - (hist[::-1] != 0).argmax()
    c, d = bins[c], bins[d]
    if dtype in ['uint8', np.uint8]:
        a, b = 0, 255
        ConStreImg = (img - c) / (d - c) * (b - a) + a
        return np.array(ConStreImg).astype('uint8')
    if dtype in ['uint16', np.uint16]:
        a, b = 0, 65535
        ConStreImg = (img - c) / (d - c) * (b - a) + a
        return np.array(ConStreImg).astype('uint16')
    if dtype is None:
        a, b = 0, 255
        ConStreImg = (img - c) / (d - c) * (b - a) + a
        return np.array(ConStreImg)


def erosion(img, k=0.8):
    marks = sorted(list(set(img.flatten())))[1:]
    eroImgs = np.zeros(img.shape, np.uint8)
    for t in marks:
        img2 = np.zeros(img.shape, np.uint8)
        img2[img == t] = 255
        _, contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        CRlst = sorted([(cv2.minEnclosingCircle(cnt), cnt)for cnt in contours], key=lambda x: x[0][1])
        while len(CRlst) > 0 and CRlst[0][0][1] < CRlst[-1][0][1] / 10:
            CRlst = CRlst[1:]
        contours = [cnt[1] for cnt in CRlst]
        for i in range(len(contours)):
            img3 = np.zeros(img.shape, np.uint8)
            cv2.drawContours(img3, contours, i, (255), cv2.FILLED)
            # _, r = cv2.minEnclosingCircle(contours[i])
            # _, _, w, h = cv2.boundingRect(contours[i])
            # r = (w + h) / 2
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            r = min([np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[0] - box[2]), np.linalg.norm(box[0] - box[3])])
            # r = min([w, h])
            d_se = int((1 - k) * r)
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d_se + 1, 2 * d_se + 1), (d_se, d_se))
            img4 = cv2.erode(img3, element)
            _, contours2, _ = cv2.findContours(img4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            CRlst = sorted([(cv2.minEnclosingCircle(cnt), cnt)for cnt in contours2], key=lambda x: x[0][1])
            while len(CRlst) > 0 and CRlst[0][0][1] < CRlst[-1][0][1] / 10:
                CRlst = CRlst[1:]
            contours2 = [cnt[1] for cnt in CRlst]
            cv2.drawContours(eroImgs, contours2, -1, (1), cv2.FILLED)
    # print(k)
    # cv2.imshow('eroImgs', contrast_stretch(eroImgs))
    # cv2.waitKey()
    return eroImgs  # .astype(np.uint8)


def compute_weights(imgs):
    results = []
    for img in imgs:
        weights = np.ones(img.shape) * 0.1
        weights += img / 255 - 0.2
        results.append(weights)
    return results


def weights_normalize(weights, min_prob=0.05, max_prob=0.95):
    self_min, self_max = np.min(weights), np.max(weights)
    return (weights - self_min) / (self_max - self_min) * (max_prob - min_prob) + min_prob


def scale_image(imgs, width=288):
    result = []
    for img in imgs:
        h, w = img.shape
        scale_factor = width / w
        height = round(h * scale_factor)
        result.append(cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST))
    return result


def scale2nearest_multiple(imgs, multi=16):
    result = []
    for img in imgs:
        h, w = img.shape
        if h % multi != 0 or w % multi != 0:
            h = h // multi
            w = w // multi
            h *= multi
            w *= multi
            result.append(cv2.resize(img, (w, h), cv2.INTER_NEAREST))
    return result


def scale2spec_size(imgs, w, h):
    return [cv2.resize(img, (w, h), cv2.INTER_NEAREST)for img in imgs]


def preprocess(img_size=512):
    # Training Data Preparation
    seqPath = {'sequences_ori/DIC-C2DH-HeLa/': ['HE', 0.8, 0.95],
               'sequences_ori/Fluo-N2DL-HeLa/': ['CS', 0.8, 0.98],
               'sequences_ori/PhC-C2DL-PSC/': ['median', 0.8, 0.98]}
    Images = []
    AugImages = []
    Masks = []
    AugMasks = []
    Markers = []
    AugMarkers = []
    EroMasks = []
    AugEroMasks = []
    for dataset, normType in seqPath.items():
        filesLst = []
        filemsLst = []
        filemrLst = []
        for i in range(1, 3):
            curSeq = dataset + '0' + str(i) + '/'
            curMask = dataset + '0' + str(i) + '_ST/SEG/'
            curMarker = dataset + '0' + str(i) + '_GT/TRA/'
            fs = os.walk(curSeq)
            fms = os.walk(curMask)
            fmr = os.walk(curMarker)
            for curdir, _, filenames in fs:
                filesLst.append((curdir, sorted(filenames)))
            for curdir, _, filenames in fms:
                filemsLst.append((curdir, sorted(filenames)))
            for curdir, _, filenames in fmr:
                filemrLst.append((curdir, sorted(filenames)))
        for i in range(2):
            for mskname in filemsLst[i][1]:
                filename = 't' + mskname[7:]
                markername = mskname[:4] + 'track' + mskname[7:]
                img = cv2.imread(filesLst[i][0] + filename, 0)
                msk = cv2.imread(filemsLst[i][0] + mskname, cv2.IMREAD_UNCHANGED)
                marker = cv2.imread(filemrLst[i][0] + markername, cv2.IMREAD_UNCHANGED)
                eroMsk = erosion(msk, normType[1])
                # print(filemsLst[i][0] + mskname)
                # cv2.imshow('marker', contrast_stretch(marker))
                # cv2.imshow('eromask', contrast_stretch(eroMsk))
                if normType[2] < 1:
                    msk = erosion(msk, normType[2])
                # cv2.imshow('mask', contrast_stretch(msk))
                # cv2.waitKey()
                [img, msk, marker, eroMsk] = scale2spec_size([img, msk, marker, eroMsk], img_size, img_size)
                msk[msk > 0] = 1
                msk = msk.astype(np.uint8)
                marker[marker > 0] = 1
                marker = marker.astype(np.uint8)
                eroMsk[eroMsk > 0] = 1
                results = data_augmentation([img, marker, msk, eroMsk])
                augImg, augMarker, augMsk, augEroMsk = results[0], results[1], results[2], results[3]
                normImg = data_normalization(img, normType[0])
                normAugImg = data_normalization(augImg, normType[0])
    #             cv2.imwrite('processed_data' + filesLst[i][0][9:-1] + ' Masks/' + mskname, msk)
    #             cv2.imwrite('processed_data' + filesLst[i][0][9:-1] + ' Masks/' + mskname[:-4] + '_aug.tif', augMsk)
    #             cv2.imwrite('processed_data' + filesLst[i][0][9:-1] + ' Masks Erosion/' + mskname, marker)
    #             cv2.imwrite('processed_data' + filesLst[i][0][9:-1] + ' Masks Erosion/' + mskname[:-4] + '_aug.tif', augMarker)
                Images.append(normImg)
                AugImages.append(normAugImg)
                Masks.append(msk)
                AugMasks.append(augMsk)
                Markers.append(marker)
                AugMarkers.append(augMarker)
                # print(dataset[14:-1] not in ['PhC-C2DL-PSC'])
                if dataset[14:-1] not in ['PhC-C2DL-PSC']:
                    EroMasks.append(eroMsk)
                    AugEroMasks.append(augEroMsk)
                else:
                    EroMasks.append(marker)
                    AugEroMasks.append(augMarker)

    Images = np.array(Images)
    AugImages = np.array(AugImages)
    Images = np.concatenate((Images, AugImages), axis=0)
    length, height, width = Images.shape
    Images = Images.reshape((length, 1, height, width))
    Masks = np.array(Masks)
    AugMasks = np.array(AugMasks)
    Masks = np.concatenate((Masks, AugMasks), axis=0)
    Masks = Masks.reshape((length, 1, height, width))
    Markers = np.array(Markers)
    AugMarkers = np.array(AugMarkers)
    Markers = np.concatenate((Markers, AugMarkers), axis=0)
    Markers = Markers.reshape((length, 1, height, width))
    EroMasks = np.array(EroMasks)
    AugEroMasks = np.array(AugEroMasks)
    EroMasks = np.concatenate((EroMasks, AugEroMasks), axis=0)
    EroMasks = EroMasks.reshape((length, 1, height, width))
    np.save('processed_data/1-2-Imgs.npy', Images)
    np.save('processed_data/1-2-Masks.npy', Masks)
    np.save('processed_data/1-2-Markers.npy', Markers)
    np.save('processed_data/1-2-EroMasks.npy', EroMasks)
    # Testing Data Preparation
    seqPath = {'DIC-C2DH-HeLa/': ['HE', 0.8, 0.95],
               'Fluo-N2DL-HeLa/': ['CS', 0.8, 0.98],
               'PhC-C2DL-PSC/': ['median', 0.8, 0.98]}
    for dataset, normType in seqPath.items():
        for i in range(1, 5):
            Images = []
            filesLst = []
            oriImages = []
            curSeq = dataset + 'Sequence ' + str(i) + '/'
            fs = os.walk(curSeq)
            for curdir, _, filenames in fs:
                filesLst.append((curdir, sorted(filenames)))
            for filename in filesLst[0][1]:
                img = cv2.imread(filesLst[0][0] + filename, 0)
                oriImages.append(contrast_stretch(img))
                [img] = scale2spec_size([img], img_size, img_size)
                normImg = data_normalization(img, normType[0])
                np.save('processed_data' + filesLst[0][0][9:] + filename[:4] + '.npy', normImg)
                Images.append(normImg)
            Images = np.array(Images)
            length, height, width = Images.shape
            Images = Images.reshape((length, 1, height, width))
            np.save('processed_data/' + dataset[:-1] + f'-{str(i)}-Imgs.npy', Images)
            oriImages = np.array(oriImages)
            np.save('processed_data/' + dataset[:-1] + f'-{str(i)}-oriImgs.npy', oriImages)


if __name__ == '__main__':
    preprocess()
