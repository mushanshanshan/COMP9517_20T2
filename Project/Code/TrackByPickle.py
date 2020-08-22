import pickle
import numpy as np
import cv2
import os
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
import time
import sys
from itertools import chain


def set_global_params(seq):
    '''
    -----------------global_params------------------

    track_cost_params:
    Weights of [ perimeter_diff , area_diff, centroid_diff ]

    new_cell_cost_thres：
    New cell cost threshold

    original_resolution:
    Resolution of original dataset

    trace_line_length:
    The length of track_line by how many frame

    max_move_dist:
    Cell moving distance threshold

    max_div_area_thres/min_div_area_thres:
    Initial threshold of separated cell area



    -------------------------------------------------
    '''
    if seq == 'DIC':
        return {
            'track_cost': [0.2, 0.1, 5],
            'new_cell_cost_thres': 13000,
            'original_resolution': [512, 512],
            'trace_line_length': 10,
            'max_move_dist': 50,
            'max_div_area_thres': 10000,
            'min_div_area_thres': 1000,
            'div_dist_thres': 100,
            'normalize': 0
        }
    if seq == 'PHC':
        return {
            'track_cost': [1, 4, 9],
            'new_cell_cost_thres': 1100,
            'original_resolution': [720, 576],
            'trace_line_length': 6,
            'max_move_dist': 160,
            'max_div_area_thres': 100,
            'min_div_area_thres': 0,
            'div_dist_thres': 30,
            'normalize': 0
        }
    if seq == 'FLU':
        return {
            'track_cost': [1, 1, 7],
            'new_cell_cost_thres': 500,
            'original_resolution': [1100, 700],
            'trace_line_length': 10,
            'max_move_dist': 150,
            'max_div_area_thres': 150,
            'min_div_area_thres': 30,
            'div_area_thres': 1500,
            'div_dist_thres': 35,
            'normalize': 1
        }


def rangelen(list):
    return range(len(list))


def track_cost(cell_1, cell_2, params):
    perimeter_diff = abs(((cell_1[2] + cell_1[3]) - (cell_2[2] + cell_2[3])) * 2) * params[0]
    area_diff = np.sqrt(abs((cell_1[2] * cell_1[3]) - (cell_2[2] * cell_2[3]))) * params[1]
    centroid_diff = (np.linalg.norm(np.array(centroid(cell_1)) - np.array(centroid(cell_2))) * params[2])

    return perimeter_diff + area_diff + centroid_diff


def track_dist(cell_1, cell_2):
    return np.linalg.norm(np.array(cell_1[:2]) - np.array(cell_2[:2]))


def pHash(img_input):
    img = deepcopy(img_input)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)

    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img

    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize((32, 32), refcheck=False)

    img_list = list(chain.from_iterable(vis1))

    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])


def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 100


def bbox_cut(img, bbox):
    bbox[2] = 1 if bbox[2] == 0 else bbox[2]
    bbox[3] = 1 if bbox[3] == 0 else bbox[3]
    new_img = img[bbox[1]:bbox[1] + bbox[3] + 1, bbox[0]: bbox[0] + bbox[2] + 1]
    return new_img


def optimize_pair(prev_frame_bbox, curr_frame_bbox, cost_mat, dis_mat, all_params):
    new_cell = []
    cost_mat = cost_mat.T
    dis_mat = dis_mat.T
    boxes = [None] * len(prev_frame_bbox)
    processed = [0] * len(curr_frame_bbox)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    col_ind = col_ind.tolist()

    for i in rangelen(col_ind):
        if cost_mat[row_ind[i], col_ind[i]] < all_params['new_cell_cost_thres'] and dis_mat[row_ind[i], col_ind[i]] < \
                all_params['max_move_dist']:
            boxes[row_ind[i]] = curr_frame_bbox[col_ind[i]]
        else:
            new_cell.append(curr_frame_bbox[col_ind[i]])
        processed[col_ind[i]] = 1

    for i in rangelen(processed):
        if processed[i] == 0:
            new_cell.append(curr_frame_bbox[i])

    return boxes, new_cell


def frame_track(prev_frame_bbox, curr_frame_bbox, prev_frame_img, curr_frame_img, prev_frame_cell_num, all_params):
    cost_mat = np.zeros((len(curr_frame_bbox), len(prev_frame_bbox)))
    dis_mat = np.zeros((len(curr_frame_bbox), len(prev_frame_bbox)))
    boxes = [None] * prev_frame_cell_num
    opencv_pred_box = opencv_track_boxes(prev_frame_img, curr_frame_img, prev_frame_bbox)

    for x in range(len(curr_frame_bbox)):
        for y in range(len(prev_frame_bbox)):

            if prev_frame_bbox[y] != None and curr_frame_bbox[x] != None:
                cost_mat[x, y] = track_cost(curr_frame_bbox[x], opencv_pred_box[y], all_params['track_cost'])
                dis_mat[x, y] = track_dist(curr_frame_bbox[x], opencv_pred_box[y])

            else:
                cost_mat[x, y] = 10 ** 9
                dis_mat[x, y] = 10 ** 9

    boxes, new_cell_box = optimize_pair(prev_frame_bbox, curr_frame_bbox, cost_mat, dis_mat, all_params)
    new_cell_num = len(new_cell_box)

    '''
    for y in range(len(prev_frame_bbox)):
        min_dis = cost_mat[:, y:y + 1].tolist()
        min_dis.sort()
        for dis in min_dis:
            dis = dis[0]
            if dis < all_params['new_cell_cost_thres']:
                x = np.where(cost_mat[:, y:y + 1] == dis)[0][0]
                if processed_boxes[x] == 0:
                    boxes[y] = curr_frame_bbox[x]
                    processed_boxes[x] = 1
                    break

    new_cell_num = 0
    new_cell_box = []
    for x in range(len(curr_frame_bbox)):
        if processed_boxes[x] == 0:
            new_cell_box.append(curr_frame_bbox[x])
            new_cell_num += 1
    '''

    return boxes, new_cell_box, prev_frame_cell_num + new_cell_num


def div_box(box_1, box_2):
    if box_1 == None or box_2 == None:
        return None, None, None, None
    x = min(box_1[0], box_2[0])
    y = min(box_1[1], box_2[1])
    max_x = max(box_1[0] + box_1[2], box_2[0] + box_2[2])
    max_y = max(box_1[1] + box_1[3], box_2[1] + box_2[3])
    return x, y, max_x, max_y


def draw_box(img, boxes, all_div_pair, all_params):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if boxes[i] != None:
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]),
                          (0, 255, 0), 1)
            cv2.putText(img, str(i), (boxes[i][0], boxes[i][1]), font, 0.5, (0, 0, 255), 1)
    for i in range(len(all_div_pair)):
        all_div_pair[i][2] -= 1
        x, y, max_x, max_y = div_box(boxes[all_div_pair[i][0]], boxes[all_div_pair[i][1]])
        if x != None and max_x - x < all_params['div_dist_thres'] * 2 and max_y - y < all_params['div_dist_thres'] * 2:
            cv2.rectangle(img, (x, y), (max_x, max_y), (255, 255, 0), 2)
        else:
            all_div_pair[i][2] = 0
    need_pop = []
    for i in range(len(all_div_pair)):
        if all_div_pair[i][2] == 0:
            need_pop.append(i)
    need_pop.sort(reverse=True)
    for i in need_pop:
        all_div_pair.pop(i)

    return img, all_div_pair


def centroid(box):
    return int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)


def draw_line(img, boxes_timeline, all_params):
    for i in range(len(boxes_timeline[-1])):
        if boxes_timeline[-1][i] is not None:
            for t in range(all_params['trace_line_length']):
                if t == len(boxes_timeline)-1 or len(boxes_timeline[-1 - t]) <= i or len(boxes_timeline[-1 - t - 1]) <= i or boxes_timeline[-1 - t][
                    i] == None:
                    break
                cv2.line(img, centroid(boxes_timeline[-1 - t][i]), centroid(boxes_timeline[-1 - t - 1][i]),
                         (255, 0, 255), 1, 4)
    return img


def scale_pickle(data_list, all_params):
    scale_x = all_params['original_resolution'][0]
    scale_y = all_params['original_resolution'][1]
    for frame in data_list:
        for i in frame:
            i[0] = 1 if i[0] == 0 else int(scale_x / 512 * i[0])
            i[2] = 1 if i[2] == 0 else int(scale_x / 512 * i[2])
            i[1] = 1 if i[1] == 0 else int(scale_y / 512 * i[1])
            i[3] = 1 if i[3] == 0 else int(scale_y / 512 * i[3])
    return data_list


def read_imgs(dir):
    img_list = []
    for filename in os.listdir(dir):
        if filename[-3:] == 'tif':
            img_list.append(cv2.imread(dir + filename))
    return img_list


def opencv_track_boxes(prev_frame, curr_frame, prev_boxes):
    multiTracker = cv2.MultiTracker_create()
    for box in prev_boxes:
        if box != None:
            multiTracker.add(cv2.TrackerKCF_create(), prev_frame, tuple(box))
    success, boxes = multiTracker.update(curr_frame)
    boxes = boxes.tolist()
    for i in range(len(prev_boxes)):
        if prev_boxes[i] == None:
            boxes.insert(i, None)

    return boxes


def box_area(box):
    return box[2] * box[3]


def check_division(prev_box, new_box, all_params):
    div_pair = []
    dist_mat = np.zeros((len(new_box), len(prev_box)))
    for x in range(len(new_box)):
        for y in range(len(prev_box)):

            if prev_box[y] != None:
                dist_mat[x, y] = np.linalg.norm(np.array(centroid(new_box[x])) - np.array(centroid(prev_box[y])))
            else:
                dist_mat[x, y] = np.inf

    min_index = np.argmin(dist_mat, axis=1).tolist()
    for i in range(len(min_index)):
        if all_params['min_div_area_thres'] < box_area(new_box[i]) < all_params['max_div_area_thres'] and \
                all_params['min_div_area_thres'] < box_area(prev_box[min_index[i]]) < all_params[
            'max_div_area_thres'] and \
                np.linalg.norm(np.array(new_box[i][:2]) - np.array(prev_box[min_index[i]][:2])) < all_params[
            'div_dist_thres'] and \
                new_box[i][0] > 20 and new_box[i][1] > 20 and new_box[i][2] < all_params['original_resolution'][
            0] - 20 and \
                new_box[i][3] < all_params['original_resolution'][1] - 20:
            div_pair.append([min_index[i], len(prev_box) + i, 5])
    return prev_box + new_box, div_pair


def normalize(image):
    img = image.copy().astype(np.float32)
    img -= np.mean(img)
    img /= np.linalg.norm(img)
    img = np.clip(img, 0, 255)
    img *= (1. / float(img.max()))
    return (img * 255).astype(np.uint8)



def track(data_list, Type, output_dir):
    print(output_dir)
    try:
        os.makedirs(output_dir[:-1])
    except:
        pass
    all_div_pair = []
    all_params = set_global_params(Type)
    data_list = scale_pickle(data_list, all_params)
    frame_1 = data_list[0]
    box_time_line = [frame_1]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(output_dir + Type + '.avi', fourcc, 5, tuple(all_params['original_resolution']),
                                  True)
    number_of_cell = []
    cal_time = []
    for i in range(len(img_list) - 1):
        print('tracking:' + str(i))
        start = time.time()

        frame_2 = data_list[i + 1]

        img_frame_1 = deepcopy(img_list[i])
        img_frame_2 = deepcopy(img_list[i + 1])
        if all_params['normalize'] == 1:
            img_frame_2 = normalize(img_frame_2)
        original_cell_num = len(frame_1)
        frame_2_box, frame_2_new_box, frame_2_cell_num = frame_track(frame_1, frame_2, img_frame_1, img_frame_2,
                                                                     original_cell_num, all_params)

        if frame_2_new_box != []:
            frame_2_box, div_pair = check_division(frame_2_box, frame_2_new_box, all_params)
            all_div_pair += div_pair
        img_frame_2_with_box, all_div_pair = draw_box(img_frame_2, frame_2_box, all_div_pair, all_params)

        cv2.putText(img_frame_2_with_box, 'Number of cells: ' + str(len(frame_2)) + '  Number of cell_div: ' + str(
            len(all_div_pair)) + '  Frame:' + str(i), (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (255, 100, 255), 1)
        number_of_cell.append(len(frame_2))

        frame_1 = frame_2_box + frame_2_new_box
        box_time_line.append(frame_1)
        img_frame_2_with_box = draw_line(img_frame_2_with_box, box_time_line, all_params)
        end = time.time()
        cal_time.append(end - start)
        cv2.imwrite(output_dir + str(i) + '.png',img_frame_2_with_box)
        videoWriter.write(img_frame_2_with_box)

    videoWriter.release()
    with open(output_dir +'box_list.pickle', 'wb') as f:
        pickle.dump(box_time_line, f)


def init_setup(argv):
    dir_dict = {
        'd' : 'DIC-C2DH-HeLa',
        'f' : 'Fluo-N2DL-HeLa',
        'p' : 'PhC-C2DL-PSC'
    }

    type_dict = {
        'd': 'DIC',
        'f': 'FLU',
        'p': 'PHC'
    }

    output = './Tracking_division_output/'+dir_dict[argv[1]]+'/Sequence '+str(argv[2])+'/'
    return type_dict[argv[1]], './Pickles/'+dir_dict[argv[1]]+'-'+str(argv[2])+'.pickle', ''+dir_dict[argv[1]]+'/Sequence '+str(argv[2])+'/', output



if __name__ == '__main__':
    Type, pickle_dir, img_dir, output_dir = init_setup(sys.argv)
    pickle_file = open(pickle_dir, 'rb')
    img_list = read_imgs(img_dir)


    data_list = pickle.load(pickle_file)

    track(data_list, Type, output_dir)
