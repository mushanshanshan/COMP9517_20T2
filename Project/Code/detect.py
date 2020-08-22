import torch
import numpy as np
# from preprocessing import contrast_stretch
from CNN import UNet
import cv2
from matplotlib import pyplot as plt
# import sys
# import imageio
import pickle
import argparse
from random import randint
# from pygifsicle import optimize
from preprocessing import contrast_stretch


def draw_histogram(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6, 3))
    ax = axes.ravel()
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Image')
    ax[0].set_axis_off()
    ax[1].plot(cdf_normalized, color='b')
    ax[1].hist(img.flatten(), 256, [0, 256], color='r')
    ax[1].set_title('Histogram')
    ax[1].legend(('cdf', 'histogram'), loc='upper left')
    plt.xlim([0, 256])
    fig.tight_layout()
    plt.show()


def watershed_marker(img):
    ret, img2 = cv2.threshold(img, 255 / 2, 255, cv2.THRESH_BINARY)
    img2, contours, hierarchy = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    CRLst = sorted([(cv2.minEnclosingCircle(cnt), cnt)for cnt in contours], key=lambda x: x[0][1], reverse=True)
    delete_ratio = 10
    while len(CRLst) > 0 and (CRLst[0][0][1] >= img.shape[0] // 2 or CRLst[0][0][1] >= img.shape[1] // 2):
        CRLst = CRLst[1:]
    while len(CRLst) > 0 and (CRLst[-1][0][1] < CRLst[0][0][1] / delete_ratio):
        CRLst.pop()
    contours = [cnt[1]for cnt in CRLst]
    img3 = np.zeros(img2.shape, dtype=np.uint8)
    cv2.drawContours(img3, contours, -1, (255), cv2.FILLED)
    return img3, contours


def pred2img(preds):
    results = []
    for pred in preds:
        pred = pred.cpu().detach().numpy()
        _, _, h, w = pred.shape
        pred = pred.reshape((h, w))
        a, b = np.min(pred), np.max(pred)
        pred = (pred - a) / (b - a) * 255
        results.append(pred.astype(np.uint8))
    return results


def preds2imgs(preds):
    preds = [pred.cpu().clone().detach().numpy() for pred in preds]
    # preds = [[(((p - np.min(p)) / (np.max(p) - np.min(p))) * 255).reshape(p.shape[1:]).astype(np.uint8)
    #           for p in pred]
    #          for pred in preds]
    preds = [[(p * 255).reshape(p.shape[1:]).astype(np.uint8)
              for p in pred]
             for pred in preds]
    return [[preds[j][i] for j in range(len(preds))]for i in range(len(preds[0]))]


def marker_controlled_watershed(imgs):
    img_marker, img_mask = imgs[0], imgs[1]
    thresh = 127
    # img_marker[img_marker < thresh] = 0
    img_mask[img_mask < thresh] = 0
    # img_mask[img_mask > thresh] = 255
    # cv2.imshow('ori_img_mask', img_mask)
    # cv2.imshow('ori_img_marker', img_marker)
    sure_bg = img_mask.copy()
    sure_bg[sure_bg > thresh] = 255
    sure_bg[sure_bg <= thresh] = 0

    img_marker, contours = watershed_marker(img_marker)
    # cv2.imshow('img_marker', img_marker)
    unknown = cv2.subtract(sure_bg, img_marker)
    ret, markers = cv2.connectedComponents(img_marker)
    markers += 1
    markers[unknown == 255] = 0
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_mask, markers)
    return markers, contours


def imgs_resize(imgs, width=512, height=512, interpolation=cv2.INTER_NEAREST):
    return [cv2.resize(img, (width, height), interpolation=interpolation) for img in imgs]


def pred_sequence_(marker, mask, device, loader, size):
    rectangular_imgs = []
    # frame = len(loader.dataset)
    # i = 0
    for data, ori_imgs in loader:
        data = torch.tensor(data).to(device, dtype=torch.float)
        pred = [None, None]
        pred[0] = marker(data)
        pred[1] = mask(data)
        ori_imgs = list(ori_imgs.cpu().detach().numpy())
        pred_imgs = preds2imgs(pred)
        for pred_img, ori_img in zip(pred_imgs, ori_imgs):
            hei, wid = ori_img.shape
            # ori_img = contrast_stretch(ori_img)
            # hist, _ = np.histogram(ori_img.flatten(), 256, [0, 256])
            # bg = np.argmax(hist)
            # ori_img[ori_img <= bg] = 0
            # ori_img = cv2.equalizeHist(ori_img)
            ori_img = cv2.resize(ori_img, size, interpolation=cv2.INTER_NEAREST)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
            seg_img, _ = marker_controlled_watershed(pred_img)
            # cv2.imshow('seg_img', contrast_stretch(seg_img))
            h, w = seg_img.shape
            seg_img2 = np.zeros((h, w, 3), np.uint8)
            cells = sorted(list(set(seg_img.flatten())))
            cellnum = 0
            for c in cells[1:]:
                img2 = np.zeros(seg_img.shape, dtype=np.uint8)
                img2[seg_img == c] = 255
                _, contour, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img2, contour, -1, (255), cv2.FILLED)
                # cv2.imshow('img2', img2)
                # cv2.waitKey()
                _, contour, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                CRLst = sorted([(cv2.minEnclosingCircle(cnt), cnt)for cnt in contour], key=lambda x: x[0][1], reverse=True)
                contour = [cnt[1]for cnt in CRLst]
                x, y, w, h = cv2.boundingRect(contour[0])
                # if False:
                if w >= seg_img.shape[1] - 2 or h >= seg_img.shape[0] - 2:
                    # print(1)
                    continue
                else:
                    r = randint(50, 255)
                    g = randint(50, 255)
                    b = randint(50, 255)
                    cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.drawContours(seg_img2, contour, 0, (b, g, r), cv2.FILLED)
                    cellnum += 1
            ori_img = cv2.resize(ori_img, (wid, hei), interpolation=cv2.INTER_LINEAR)
            seg_img2 = cv2.resize(seg_img2, (wid, hei), interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(f'frames/t' + str(1000 + i)[1:] + '.jpg', ori_img)
            # i += 1
            # cv2.imshow('seg_img', seg_img2)
            ori_img = np.hstack((ori_img, seg_img2))
            cv2.putText(ori_img, 'Detected Cell Num: {:4d}'.format(cellnum), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # cv2.imwrite(f'frames/t' + str(1000 + i)[1:] + '.jpg', ori_img)
            # i += 1
            # cv2.imshow('rectangle', ori_img)
            # cv2.waitKey()
            rectangular_imgs.append(ori_img)
    return rectangular_imgs


def pred_sequence(mask, marker, data, ori_img, device, seq_name, batch_size, duration):
    img_size = 512
    data = np.load(data)
    ori_img = np.load(ori_img)
    data = torch.tensor(data)
    ori_img = torch.tensor(ori_img)
    dataset = torch.utils.data.TensorDataset(data, ori_img)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rectangular_imgs = pred_sequence_(marker, mask, device, loader, (img_size, img_size))
    h, w, d = rectangular_imgs[0].shape
    out = cv2.VideoWriter(seq_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
    for i in range(len(rectangular_imgs)):
        out.write(rectangular_imgs[i])
    out.release()


def record_coordinates_(marker, mask, device, loader):
    coordinates_imgs = []
    for data, _ in loader:
        data = torch.tensor(data).to(device, dtype=torch.float)
        pred = [None, None]
        pred[0] = marker(data)
        pred[1] = mask(data)
        pred_imgs = preds2imgs(pred)
        for pred_img in pred_imgs:
            coordinates_img = []
            seg_img, _ = marker_controlled_watershed(pred_img)
            # cv2.imshow('seg_img', contrast_stretch(seg_img))
            cells = sorted(list(set(seg_img.flatten())))
            for c in cells[1:]:
                img2 = np.zeros(seg_img.shape, dtype=np.uint8)
                img2[seg_img == c] = 255
                _, contour, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img2, contour, -1, (255), cv2.FILLED)
                # cv2.imshow('img2', img2)
                # cv2.waitKey()
                _, contour, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                CRLst = sorted([(cv2.minEnclosingCircle(cnt), cnt)for cnt in contour], key=lambda x: x[0][1], reverse=True)
                contour = [cnt[1]for cnt in CRLst]
                x, y, w, h = cv2.boundingRect(contour[0])
                if w >= seg_img.shape[1] - 5 or h >= seg_img.shape[0] - 5:
                    continue
                else:
                    coordinates_img.append([x, y, w, h])
                    # print([x, y, w, h])
            coordinates_imgs.append(coordinates_img)
            # print()
    return coordinates_imgs
    # np.set_printoptions(threshold=np.inf)


def record_coordinates(mask, marker, data, device, seq_name, batch_size):
    data = np.load(data)
    ori_img = np.zeros((data.shape[0], 1))
    data = torch.tensor(data)
    ori_img = torch.tensor(ori_img)
    dataset = torch.utils.data.TensorDataset(data, ori_img)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    coordinates = record_coordinates_(marker, mask, device, loader)
    with open(seq_name + '.pickle', mode='wb')as fp:
        pickle.dump(coordinates, fp)


def pred_single_img(mask, marker, data, ori_img, device):
    img_size = 512
    data = np.load(data)
    ori_img = cv2.imread(ori_img)
    hei, wid, _ = ori_img.shape
    ori_img = cv2.resize(ori_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    h, w = data.shape
    data = data.reshape((-1, 1, h, w))
    data = torch.tensor(data).to(device, dtype=torch.float)
    pred = [None, None]
    pred[0] = marker(data)
    pred[1] = mask(data)
    pred_imgs = pred2img(pred)
    seg_img, _ = marker_controlled_watershed(pred_imgs)
    # cv2.imshow('seg_img', contrast_stretch(seg_img))
    seg_img2 = np.zeros((h, w, 3), np.uint8)
    cells = sorted(list(set(seg_img.flatten())))
    cellnum = 0
    for c in cells[1:]:
        img2 = np.zeros(seg_img.shape, dtype=np.uint8)
        img2[seg_img == c] = 255
        _, contour, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img2, contour, -1, (255), cv2.FILLED)
        _, contour, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        CRLst = sorted([(cv2.minEnclosingCircle(cnt), cnt)for cnt in contour], key=lambda x: x[0][1], reverse=True)
        contour = [cnt[1]for cnt in CRLst]
        x, y, w, h = cv2.boundingRect(contour[0])
        if w >= img_size / 1.5 or h >= img_size / 1.5:
            continue
        else:
            r = randint(50, 255)
            g = randint(50, 255)
            b = randint(50, 255)
            cv2.rectangle(ori_img, (x, y), (x + w, y + h), (b, g, r), 2)
            cv2.drawContours(seg_img2, contour, 0, (b, g, r), cv2.FILLED)
            cellnum += 1
    ori_img = cv2.resize(ori_img, (wid, hei), interpolation=cv2.INTER_LINEAR)
    seg_img2 = cv2.resize(seg_img2, (wid, hei), interpolation=cv2.INTER_LINEAR)
    ori_img = np.hstack((ori_img, seg_img2))
    cv2.putText(ori_img, 'Detected Cell Num: {:4d}'.format(cellnum), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('rectangle', ori_img)
    cv2.waitKey()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default='seq', help='single, sequence or coordinates')
    parser.add_argument('--seq', type=str, default='d4', help='d1~4,f1~4 or p1~4')
    parser.add_argument('--img', type=str, default='d3 0', help='input image num')
    parser.add_argument('--mask', type=str, default='models/mask.pkl', help='mask model')
    parser.add_argument('--marker', type=str, default='models/marker.pkl', help='marker model')
    parser.add_argument('--batch', type=int, default=4, help='batch size for predicting sequences or coordinates')
    parser.add_argument('--step', type=float, default=0.3, help='time step for sequence')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    marker = torch.load(args.marker, map_location=device)
    mask = torch.load(args.mask, map_location=device)

    if args.pred in ['single', 'img']:
        data = 'processed_data/'
        ori_img = 'sequences/'
        img = args.img.split()
        if img[0][0] in ['D', 'd']:
            data += 'DIC-C2DH-HeLa/'
            ori_img += 'DIC-C2DH-HeLa/'
        elif img[0][0] in ['F', 'f']:
            data += 'Fluo-N2DL-HeLa/'
            ori_img += 'Fluo-N2DL-HeLa/'
        elif img[0][0] in ['P', 'p']:
            data += 'PhC-C2DL-PSC/'
            ori_img += 'PhC-C2DL-PSC/'
        data += 'Sequence ' + img[0][-1] + '/t' + str(int(img[1]) + 1000)[1:] + '.npy'
        ori_img += 'Sequence ' + img[0][-1] + '/t' + str(int(img[1]) + 1000)[1:] + '.tif'
        pred_single_img(mask, marker, data, ori_img, device)
    elif args.pred in ['sequence', 'seq', 'imgs']:
        data = 'processed_data/'
        ori_img = 'processed_data/'
        if args.seq[0] in ['D', 'd']:
            data += 'DIC-C2DH-HeLa-' + args.seq[-1] + '-Imgs.npy'
            ori_img += 'DIC-C2DH-HeLa-' + args.seq[-1] + '-oriImgs.npy'
            seq_name = 'mp4/DIC-C2DH-HeLa-' + args.seq[-1]
        elif args.seq[0] in ['F', 'f']:
            data += 'Fluo-N2DL-HeLa-' + args.seq[-1] + '-Imgs.npy'
            ori_img += 'Fluo-N2DL-HeLa-' + args.seq[-1] + '-oriImgs.npy'
            seq_name = 'mp4/Fluo-N2DL-HeLa-' + args.seq[-1]
        elif args.seq[0] in ['P', 'p']:
            data += 'PhC-C2DL-PSC-' + args.seq[-1] + '-Imgs.npy'
            ori_img += 'PhC-C2DL-PSC-' + args.seq[-1] + '-oriImgs.npy'
            seq_name = 'mp4/PhC-C2DL-PSC-' + args.seq[-1]
        pred_sequence(mask, marker, data, ori_img, device, seq_name, args.batch, args.step)
    elif args.pred in ['coordinates', 'coor', 'cartisian', 'co']:
        data = 'processed_data/'
        if args.seq[0] in ['D', 'd']:
            data += 'DIC-C2DH-HeLa-' + args.seq[-1] + '-Imgs.npy'
            seq_name = 'Pickles/DIC-C2DH-HeLa-' + args.seq[-1]
        elif args.seq[0] in ['F', 'f']:
            data += 'Fluo-N2DL-HeLa-' + args.seq[-1] + '-Imgs.npy'
            seq_name = 'Pickles/Fluo-N2DL-HeLa-' + args.seq[-1]
        elif args.seq[0] in ['P', 'p']:
            data += 'PhC-C2DL-PSC-' + args.seq[-1] + '-Imgs.npy'
            seq_name = 'Pickles/PhC-C2DL-PSC-' + args.seq[-1]
        record_coordinates(mask, marker, data, device, seq_name, args.batch)


if __name__ == '__main__':
    main()
