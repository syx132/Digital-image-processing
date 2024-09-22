import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None
method = "rigid"

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    MLS变形
    Return
    ------
        A deformed image.
    """
    class MLS_deformation:
        def __init__(self, p: np.array, q: np.array, alpha = 1.0):
            self._p = p.reshape([-1,2])
            self._q = q.reshape([-1,2])
            self._alpha = alpha
            self._inv_w = lambda v: ((v[:,:1] - self._p[:,0:1].T) ** 2 + (v[:,1:] - self._p[:,1:2].T) ** 2) ** self._alpha

        def __call__(self, v):
            global method
            inv_weight = self._inv_w(v)
            weight = 1.0 / inv_weight
            zero_index_v, zero_index_i = np.nonzero(inv_weight == 0)

            p_star = weight @ self._p / np.expand_dims(weight.sum(1), 1)
            q_star = weight @ self._q / np.expand_dims(weight.sum(1), 1)
            p_hat = np.expand_dims(self._p, 0) - np.expand_dims(p_star, 1)
            q_hat = np.expand_dims(self._q, 0) - np.expand_dims(q_star, 1)

            target_fv = np.zeros_like(v)
            if method == "affine":
                lhs_M = (np.expand_dims(p_hat, 3) @ np.expand_dims(p_hat, 2) * np.expand_dims(weight, (2,3))).sum(1)
                rhs_M = (np.expand_dims(q_hat, 3) @ np.expand_dims(p_hat, 2) * np.expand_dims(weight, (2,3))).sum(1)
                M = np.linalg.solve(lhs_M, rhs_M)
                T = q_star - (np.expand_dims(p_star,1) @ M).squeeze(1)
                target_fv = (np.expand_dims(v,1) @ M).squeeze(1) + T
            else:
                A_i = np.expand_dims(weight, (2, 3)) * \
                      np.stack((p_hat, np.concatenate((p_hat[..., 1:], -p_hat[..., :1]), 2)), 2) @ \
                      np.stack((np.expand_dims(v - p_star, 1),
                                np.expand_dims(np.concatenate(((v - p_star)[:, 1:], (p_star - v)[:, :1]), 1), 1)), 3)
                if method == "similiar":
                    miu_s = (weight * (p_hat[...,0]**2 + p_hat[...,1]**2)).sum(1)
                    target_fv = (np.expand_dims(q_hat, 2) @ A_i).squeeze(2).sum(1) / np.expand_dims(miu_s, 1) + q_star
                if method == "rigid":
                    fv_hat = (np.expand_dims(q_hat, 2) @ A_i).squeeze(2).sum(1)
                    target_fv = np.expand_dims(np.linalg.norm(v - p_star,2,1)/np.linalg.norm(fv_hat,2,1),1) * fv_hat + q_star

            target_fv[zero_index_v, :] = self._p[zero_index_i, :]
            target_fv[zero_index_v, :] = self._q[zero_index_i, :]
            return target_fv

    warped_image = np.array(image)
    ### FILL: 基于MLS 实现 image warping
    rows, cols, _ = image.shape
    id_rows = np.arange(0, rows)[:,np.newaxis] + np.zeros([rows, cols])
    id_cols = np.arange(0, cols)[np.newaxis,:] + np.zeros([rows, cols])

    p = np.concatenate((source_pts[:,1:2], source_pts[:,:1]), 1)
    q = np.concatenate((target_pts[:, 1:2], target_pts[:,:1]), 1)
    func = MLS_deformation(q, p, alpha)

    v = np.concatenate((id_rows.reshape([-1,1]), id_cols.reshape([-1,1])), 1)
    orig_v = func(v)
    orig_rows, orig_cols = orig_v[:,0].reshape([rows,cols]), orig_v[:,1].reshape([rows,cols])
    t_rows, t_cols = np.round(orig_rows).astype(int), np.round(orig_cols).astype(int)
    ileg = (t_rows < 0) | (t_rows >= rows) | (t_cols < 0) | (t_cols >= cols)
    t_rows[ileg], t_cols[ileg] = 0, 0
    warped_image = image[t_rows, t_cols, :]
    warped_image[ileg] = 255
    return warped_image

def method_change(choice):
    global method
    method = choice
    return run_warping()


def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=400)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    method_radio = gr.Radio(['rigid', 'similiar', 'affine'], value='rigid')
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 改变算法
    method_radio.change(method_change, method_radio, result_image)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
