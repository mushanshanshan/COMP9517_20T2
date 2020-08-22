import cv2
import numpy as np
import sys
import pickle

def center(location_list):
    x = location_list[0]
    y = location_list[1]
    w = location_list[2]
    h = location_list[-1]
    center = [x+w//2, y+h//2]
    return center

def Euclidean_Distance(a, b):
    vec1 = np.array(a)
    vec2 = np.array(b)
    distance= np.sqrt(np.sum(np.square(vec1-vec2)))
    return distance



def full_dir(argv):
    dir_dict = {
        'd' : 'DIC-C2DH-HeLa',
        'f' : 'Fluo-N2DL-HeLa',
        'p' : 'PhC-C2DL-PSC'
    }

    return './Tracking_division_output/'+ dir_dict[argv[1].lower()[:1]] +'/Sequence '+ str(argv[2]) + '/'



def read_imgs(dir, len):
    img_list = []
    for i in range(len):
        img_list.append(cv2.imread(dir + str(i) + '.png'))
    return img_list

def load_data(dir):
    with open(dir + 'box_list.pickle','rb') as f:
        box_list = pickle.load(f)
    img_list = read_imgs(dir, len(box_list))
    return box_list, img_list


def nearest_box(img_index, box_list, ROI):
    dist = []
    for i in box_list[img_index]:
        if i != None:
            dist.append(Euclidean_Distance(i[:2], ROI[:2]))
        else:
            dist.append(np.inf)

    return np.argmin(dist)


def speed(box_list, img_index, selected_cell):
    if img_index == 0 or box_list[img_index][selected_cell] == None or box_list[img_index-1][selected_cell] == None:
        return 'This is the first time point of a cell\'s trajectory, so no speed estimate can be computed.'
    else:
        return str(Euclidean_Distance(center(box_list[img_index][selected_cell]), center(box_list[img_index-1][selected_cell])))


def total_distance(box_list, img_index, selected_cell):
    sum_dis = 0
    for i in range(1, img_index):
        if len(box_list[i-1])>selected_cell and len(box_list[i])>selected_cell and \
                box_list[i-1][selected_cell] != None and box_list[i][selected_cell] != None:
            sum_dis += Euclidean_Distance(center(box_list[i][selected_cell]), center(box_list[i-1][selected_cell]))
    return sum_dis


def net_distance(box_list, img_index, selected_cell):
    start, end = None, None
    for i in range(0, img_index):
        if len(box_list[i])>selected_cell:
            if start == None:
                start = box_list[i][selected_cell]
            end = box_list[i][selected_cell]
    return Euclidean_Distance(center(start), center(end))

def analysis(box_list, img_list, wait_time = 500):

    selected_cell = None


    for img_index in range(len(img_list)):

        if selected_cell != None:
            print('\n---------------------------------------')
            if box_list[img_index][selected_cell] == None:
                print('\n********The selected is disappered, Please select a new cell!********\n')
                selected_cell = None
            else:
                t_dis = total_distance(box_list, img_index, selected_cell)
                n_dis = net_distance(box_list, img_index, selected_cell)
                print('Analysis frame ' + str(img_index) + ', cell number ' + str(selected_cell) + ':')
                print('The speed ' + speed(box_list, img_index, selected_cell) + ' pixal per frame.')
                print('The total distance is ' + str(t_dis) + ' pixal.')
                print('The net distance is ' + str(n_dis) + ' pixal.')
                print('The confinement ratio is ' + str(t_dis/n_dis) + '.')


        cv2.imshow('Cell_analysis', img_list[img_index])

        key = cv2.waitKey(wait_time)

        if key == ord('n'):
            continue
        if key == ord('s'):
            ROI = cv2.selectROI('Please select a cell', img_list[img_index])
            selected_cell = nearest_box(img_index, box_list, ROI)
            print(selected_cell)
            cv2.destroyWindow('Please select a cell')
        if key == ord('q'):
            exit(0)
        if key == ord('d'):
            selected_cell = None
            print('\n********Deselected cells!********\n')

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Please enter the correct parameters!\n Parameter explanation : python3 Analysis.py [DATASET_NAME] [SEQ_NUMBER] [WAIT_TIME]')
        exit(0)
    print('*************Instructions*************\n')
    print('Press N to display the next frame immediately\nPress S to select a cell\nPress D to cancel analysis of selected cells\nPress Q to exit the program\n')
    dir = full_dir(sys.argv)

    box_list, img_list = load_data(dir)

    analysis(box_list, img_list, int(sys.argv[3]))










