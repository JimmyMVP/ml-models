import numpy as np

"""
Voxelize the pointcloud
Limits are limits on one side of the cloud because the pointcloud,
because the pointcloud can habe negative values. for example 25 means 25 meters on one side, 50 meters on both sides.
"""
def voxel_grid(data, res=(32.0,32.0,32.0), limits=(25,25,25)):
    
    X_RES = res[0]
    Y_RES = res[1]
    Z_RES = res[2]
    limit = limits[0]
    voxel_grid = np.zeros(shape=(X_RES, Y_RES, Z_RES))

    for p in data:

        x,y,z = p[0],p[1],p[2]
        if abs(x) > limits[0]:
            continue
        elif abs(y) > limits[1]:
            continue
        elif abs(z) > limits[2]:
            continue


        x,y,z = p[0] + limits[0]-1,p[1] + limits[1] -1 ,p[2] + limits[2]-1

        grid_x = int(np.ceil(((x) / (2*limits[0]))*X_RES))
        grid_y = int(np.ceil(((y) / (2*limits[1]))*Y_RES))
        grid_z = int(np.ceil(((z)/ (2*limits[2])*Z_RES)))

        voxel_grid[grid_x][grid_y][grid_z]+=1.0

    return voxel_grid




def tf_confusion_metrics(model, actual_classes, session, feed_dict):
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(actual_classes, 1)

  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)

  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op], 
      feed_dict
    )

  tpr = float(tp)/(float(tp) + float(fn))
  fpr = float(fp)/(float(tp) + float(fn))

  accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

  recall = tpr
  precision = float(tp)/(float(tp) + float(fp))
  
  f1_score = (2 * (precision * recall)) / (precision + recall)

  return {
    "precision" : precision,
    "f1" : f1_score,
    "recall" : recall,
    "accuracy" : accuracy
  }