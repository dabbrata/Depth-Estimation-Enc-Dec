import numpy as np
from sklearn.metrics import r2_score

r2_scores = []

# Load test data
def load_test_data():    
    print('Loading test data...', end='')
    
    rgb = np.load('/kaggle/working/eigen_test_rgb.npy')
    depth = np.load('/kaggle/working/eigen_test_depth.npy')
    crop = np.load('/kaggle/working/eigen_test_crop.npy')
    print('Test data loaded.\n')
    return rgb, depth, crop

def DepthNorm(x, maxDepth):
    return maxDepth / x

def predict(model, images, minDepth=10, maxDepth=1000, batch_size=6):
    # Support multiple RGBs, one RGB image, even grayscale 
   
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    images = tf.image.resize(images, [240, 320])
    predictions = model.predict(images, batch_size=batch_size)
    return predictions

def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []    
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )
    return np.stack(scaled)

def evaluate(model, rgb, depth, crop, batch_size=6):
    def compute_errors(gt, pred):

        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        gt_pool = np.expand_dims(np.expand_dims(gt, axis=0), axis=-1)  # Shape: (1, H, W, 1)
        pred_pool = np.expand_dims(np.expand_dims(pred, axis=0), axis=-1)  # Shape: (1, H, W, 1)
        gt_pool = AveragePooling2D(pool_size=(7, 7), strides=(2, 2))(gt_pool)
        pred_pool = AveragePooling2D(pool_size=(7, 7), strides=(2, 2))(pred_pool)
        gt_pool = np.squeeze(gt_pool)  # Shape: (H_new, W_new)
        pred_pool = np.squeeze(pred_pool)  # Shape: (H_new, W_new)
        # Flatten the depth maps
        ground_truth_flat = gt_pool.flatten()
        predicted_flat = pred_pool.flatten()

        # Compute R^2 score for the current image and store it
        r2 = r2_score(ground_truth_flat, predicted_flat)
        r2_scores.append(r2)
        thresh = np.maximum((gt / pred), (pred / gt))
        
        a1 = (thresh < 1.25   ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean((np.abs(gt - pred) / gt))
        
        rmse = (gt - pred) ** 2 
        rmse = np.sqrt(rmse.mean())

        log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()

        return a1, a2, a3, abs_rel, rmse, log_10

    depth_scores = np.zeros((6, len(rgb))) # six metrics

    bs = batch_size

    for i in range(len(rgb)//bs):    #len(rgb)//bs
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]
        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]

        pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
#   
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            errors = compute_errors((true_y[j]), (0.75 * pred_y[j]))

            for k in range(len(errors)):
                depth_scores[k][(i*bs)+j] = errors[k]

    e = depth_scores.mean(axis=1)

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))



rgb, depth, crop = load_test_data() 
evaluate(model, rgb, depth, crop)

