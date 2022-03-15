import tensorflow.compat.v1 as tf

def graph_visualizer(graph_path):
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.get_default_graph()
    tf.import_graph_def(graph_def, name='TfPoseEstimator')
    sess = tf.Session(graph=graph)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs", sess.graph)
        writer.close()

if __name__ == "__main__":
    graph_visualizer("./mars-small128.pb")
