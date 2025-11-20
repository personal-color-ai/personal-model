import argparse
import glob
import os
import os.path as osp
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.filters import gaussian

import facer
# from model import BiSeNet


def get_rgb_codes(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    seg_probs = seg_probs.cpu() #if you are using GPU

    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    llip = tensor[:, :, 7]
    ulip = tensor[:,:,9]
    lips = llip+ulip
    binary_mask = (lips >= 0.5).astype(int)

    sample = cv2.imread(path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    indices = np.argwhere(binary_mask)   #binary mask location extraction
    rgb_codes = img[indices[:, 0], indices[:, 1], :] #RGB color extraction by pixels
    return rgb_codes


def get_eye_rgb_codes(path):
    """눈 영역의 RGB 코드 추출 (노이즈 필터링 포함)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    seg_probs = seg_probs.cpu()  # if you are using GPU

    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    # 눈 영역 추출 (인덱스 4: re, 인덱스 5: le)
    re = tensor[:, :, 4]  # right eye
    le = tensor[:, :, 5]  # left eye
    eyes = re + le
    binary_mask = (eyes >= 0.5).astype(np.uint8)
    
    # 노이즈 제거: 연결된 컴포넌트 중 가장 큰 두 개만 선택 (실제 눈 두 개)
    # stats 구조: [left, top, width, height, area]
    # 인덱스 4가 area (픽셀 개수)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels > 1:  # 배경 + 최소 1개 이상의 컴포넌트
        # 배경 제외하고 크기(area) 기준 정렬 (배경은 인덱스 0)
        component_sizes = [(i, stats[i, 4]) for i in range(1, num_labels)]  # stats[i, 4] = area
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 가장 큰 두 개의 컴포넌트만 선택 (양쪽 눈)
        filtered_mask = np.zeros_like(binary_mask)
        for idx, _ in component_sizes[:2]:  # 상위 2개만
            filtered_mask[labels == idx] = 1
        
        binary_mask = filtered_mask

    sample = cv2.imread(path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    indices = np.argwhere(binary_mask)  # binary mask location extraction
    rgb_codes = img[indices[:, 0], indices[:, 1], :]  # RGB color extraction by pixels
    return rgb_codes

def filter_lip_random(rgb_codes,randomNum=40):
    blue_condition = (rgb_codes[:, 2] <= 227)
    red_condition = (rgb_codes[:, 0] >= 97)
    filtered_rgb_codes = rgb_codes[blue_condition & red_condition]
    random_index = np.random.randint(0,filtered_rgb_codes.shape[0],randomNum)
    random_rgb_codes = filtered_rgb_codes[random_index]
    return random_rgb_codes


def calc_dis(rgb_codes):
    spring = [[253,183,169],[247,98,77],[186,33,33]]
    summer = [[243,184,202],[211,118,155],[147,70,105]]
    autum = [[210,124,110],[155,70,60],[97,16,28]]
    winter = [[237,223,227],[177,47,57],[98,14,37]]
  
    res = []
    for i in range(len(rgb_codes)):
      sp = np.inf
      su = np.inf
      au = np.inf
      win = np.inf
      for j in range(3):
        sp = min(sp, np.linalg.norm(rgb_codes[i] - spring[j]))
        su = min(su, np.linalg.norm(rgb_codes[i]- summer[j]))
        au = min(au, np.linalg.norm(rgb_codes[i] - autum[j]))
        win = min(win, np.linalg.norm(rgb_codes[i] - winter[j]))
    
      min_type = min(sp, su, au, win)
      if min_type == sp:
        ctype = "sp"
      elif min_type == su:
        ctype = "su"
      elif min_type == au:
        ctype = "au"
      elif min_type == win:
        ctype = "win"
    
      res.append(ctype)
    return res


def save_skin_mask(img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)  # image: 1 x 3 x h x w
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    with torch.inference_mode():
      faces = face_detector(image)

    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
      faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu() #if you are using GPU
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    face_skin = tensor[:, :, 1]
    binary_mask = (face_skin >= 0.5).astype(int)

    sample = cv2.imread(img_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    masked_image = np.zeros_like(img) 
    try: 
      masked_image[binary_mask == 1] = img[binary_mask == 1] 
      masked_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2RGB)
      cv2.imwrite("temp.jpg" , masked_image)
    except:
      print("error occurred")


def visualize_masks(img_path, output_path="mask_visualization.jpg"):
    """
    눈, 입술, 피부 영역을 시각화하여 이미지로 저장
    
    Args:
        img_path: 원본 이미지 경로
        output_path: 시각화 이미지 저장 경로
    
    Returns:
        시각화된 이미지 배열 (RGB 형식)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    
    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu()
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    # 원본 이미지 로드
    sample = cv2.imread(img_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    
    # 마스크 생성
    # 피부 (인덱스 1)
    face_skin = tensor[:, :, 1]
    skin_mask = (face_skin >= 0.5).astype(float)
    
    # 입술 (인덱스 7: ulip, 인덱스 9: llip)
    ulip = tensor[:, :, 7]
    llip = tensor[:, :, 9]
    lip_mask = ((ulip + llip) >= 0.5).astype(float)
    
    # 눈 (인덱스 4: re, 인덱스 5: le)
    re = tensor[:, :, 4]
    le = tensor[:, :, 5]
    eyes = re + le
    eye_binary_mask = (eyes >= 0.5).astype(np.uint8)
    
    # 노이즈 제거: 연결된 컴포넌트 중 가장 큰 두 개만 선택 (실제 눈 두 개)
    # stats 구조: [left, top, width, height, area]
    # 인덱스 4가 area (픽셀 개수)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eye_binary_mask, connectivity=8)
    
    if num_labels > 1:  # 배경 + 최소 1개 이상의 컴포넌트
        # 배경 제외하고 크기(area) 기준 정렬
        component_sizes = [(i, stats[i, 4]) for i in range(1, num_labels)]  # stats[i, 4] = area
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 가장 큰 두 개의 컴포넌트만 선택 (양쪽 눈)
        filtered_eye_mask = np.zeros_like(eye_binary_mask)
        for idx, _ in component_sizes[:2]:  # 상위 2개만
            filtered_eye_mask[labels == idx] = 1
        
        eye_mask = filtered_eye_mask.astype(float)
    else:
        eye_mask = eye_binary_mask.astype(float)
    
    # 시각화용 이미지 생성
    vis_img = img.copy().astype(np.float32)
    
    # 각 영역을 다른 색으로 오버레이 (반투명)
    alpha = 0.5  # 투명도
    
    # 피부 영역: 노란색 (255, 255, 0)
    skin_overlay = np.zeros_like(vis_img)
    skin_overlay[:, :, 0] = 255  # R
    skin_overlay[:, :, 1] = 255  # G
    skin_overlay[:, :, 2] = 0    # B
    vis_img = vis_img * (1 - skin_mask[:, :, np.newaxis] * alpha) + skin_overlay * (skin_mask[:, :, np.newaxis] * alpha)
    
    # 입술 영역: 빨간색 (255, 0, 0)
    lip_overlay = np.zeros_like(vis_img)
    lip_overlay[:, :, 0] = 255  # R
    lip_overlay[:, :, 1] = 0    # G
    lip_overlay[:, :, 2] = 0    # B
    vis_img = vis_img * (1 - lip_mask[:, :, np.newaxis] * alpha) + lip_overlay * (lip_mask[:, :, np.newaxis] * alpha)
    
    # 눈 영역: 파란색 (0, 0, 255)
    eye_overlay = np.zeros_like(vis_img)
    eye_overlay[:, :, 0] = 0    # R
    eye_overlay[:, :, 1] = 0    # G
    eye_overlay[:, :, 2] = 255  # B
    vis_img = vis_img * (1 - eye_mask[:, :, np.newaxis] * alpha) + eye_overlay * (eye_mask[:, :, np.newaxis] * alpha)
    
    # uint8로 변환
    vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
    
    # 저장
    try:
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_img_bgr)
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    return vis_img