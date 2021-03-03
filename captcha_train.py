# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
import sys

import tensorflow as tf
import captcha_model as captcha

FLAGS = None


def run_train():
    """Train CAPTCHA for a number of steps."""

    with tf.Graph().as_default():
        # 传入标准参数
        #
        # 权重w 0.1；偏置量 bias 0.1
        # SGD 优化器 AdamOptimizer，学习率 0.0001，keep_prob=0.5

        images, labels = captcha.inputs(train=True, batch_size=FLAGS.batch_size)

        logits = captcha.inference(images, keep_prob=0.5)

        loss = captcha.loss(logits, labels)

        train_op = captcha.training(loss)

        saver = tf.train.Saver(tf.global_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 每 10 step 统计 loss，< 0.1 终止训练防止过度拟合
        # 每 100 step 更新1次网络参数
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step % 10 == 0:
                    print('>> Step %d run_train: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                                            duration))
                if step % 100 == 0:
                    print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
                    saver.save(sess, FLAGS.checkpoint, global_step=step)
                step += 1
        except Exception as e:
            print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
            saver.save(sess, FLAGS.checkpoint, global_step=step)
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    run_train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./captcha_train',
        help='Directory where to write event logs.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./captcha_train/captcha',
        help='Directory where to write checkpoint.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
