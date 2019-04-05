import tensorflow as tf
import numpy as np
import functools

from pathlib import Path

from tensorflow.core.protobuf import saved_model_pb2, meta_graph_pb2
from google.protobuf import message, text_format

dir(tf.contrib)  # contrib ops lazily loaded


def parse_saved_model(export_dir):
    """
    Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.
        Args:
            export_dir: Directory containing the SavedModel file.
        Returns:
            A `SavedModel` protocol buffer.
        Raises:
            IOError: If the file does not exist, or cannot be successfully parsed.
    """
    # Build the path to the SavedModel in pbtxt format.
    path_to_pbtxt = Path(export_dir).joinpath(
        tf.saved_model.constants.SAVED_MODEL_FILENAME_PBTXT
    )
    # Build the path to the SavedModel in pb format.
    path_to_pb = Path(export_dir).joinpath(
        tf.saved_model.constants.SAVED_MODEL_FILENAME_PB
    )

    # Parse the SavedModel protocol buffer.
    saved_model = saved_model_pb2.SavedModel()
    if path_to_pb.is_file():
        try:
            file_content = path_to_pb.read_bytes()
            saved_model.ParseFromString(file_content)
            return saved_model
        except message.DecodeError as e:
            raise IOError("Cannot parse file %s: %s." % (path_to_pb, str(e)))
    elif path_to_pbtxt.is_file():
        try:
            file_content = path_to_pbtxt.read_bytes()
            text_format.Merge(file_content.decode("utf-8"), saved_model)
            return saved_model
        except text_format.ParseError as e:
            raise IOError("Cannot parse file %s: %s." % (path_to_pbtxt, str(e)))
    else:
        raise IOError(
            "SavedModel file does not exist at: %s/{%s|%s}"
            % (
                export_dir,
                tf.saved_model.constants.SAVED_MODEL_FILENAME_PBTXT,
                tf.saved_model.constants.SAVED_MODEL_FILENAME_PB,
            )
        )


def load_saved_model(export_dir, tags):
    saved_model = parse_saved_model(export_dir)
    for meta_graph_def in saved_model.meta_graphs:
        if set(meta_graph_def.meta_info_def.tags) == set(tags):
            return meta_graph_def
    raise IOError("Couldn't load saved_model")


class Model:

    default_tag = tf.saved_model.tag_constants.SERVING

    def __init__(self, path, tags, loop, sess_args=None):
        if sess_args is None:
            sess_args = {}

        # load ConfigProto from saved model
        try:
            meta_graph_def = load_saved_model(path, tags)
            config = tf.ConfigProto()
            meta_graph_def.meta_info_def.any_info.Unpack(config)
        except Exception:
            raise IOError("Couldn't load saved_model")

        if "config" in sess_args:
            config.MergeFrom(sess_args["config"])
        sess_args["config"] = config

        self.sess = tf.Session(**sess_args)
        self.loop = loop
        try:
            self.graph_def = tf.saved_model.loader.load(self.sess, tags, path)
        except Exception:
            raise IOError("Couldn't load saved_model")

    async def parse(self, method, request, validate_batch):
        signature = self.graph_def.signature_def[method]
        inputs = signature.inputs
        outputs = signature.outputs

        query_params = {}
        batch_length = 0
        for key, value in inputs.items():
            if key not in request:
                raise ValueError(
                    "Request missing required key %s for method %s" % (key, method)
                )

            input_json = request[key]

            # input_json = list(map(base64.b64decode, input_json))
            dtype = tf.as_dtype(inputs[key].dtype).as_numpy_dtype
            try:
                tensor = np.asarray(input_json, dtype=dtype)
            except ValueError as e:
                raise ValueError("Incompatible types for key %s: %s" % (key, e))
            correct_shape = tf.TensorShape(inputs[key].tensor_shape)
            input_shape = tf.TensorShape(tensor.shape)
            if not correct_shape.is_compatible_with(input_shape):
                raise ValueError(
                    "Shape of input %s %s not compatible with %s"
                    % (key, input_shape.as_list(), correct_shape.as_list())
                )
            if validate_batch:
                try:
                    if batch_length > 0 and batch_length != input_shape.as_list()[0]:
                        raise ValueError("The outer dimension of tensors did not match")
                    batch_length = input_shape.as_list()[0]
                except IndexError:
                    raise ValueError("%s is a scalar and cannot be batched" % key)
            query_params[value.name] = tensor

        result_params = {
            key: self.sess.graph.get_tensor_by_name(val.name)
            for key, val in outputs.items()
        }

        return query_params, result_params

    async def query(self, query_params, result_params):
        """ TODO: Interface via FIFO queue """
        return await self.loop.run_in_executor(
            None,
            functools.partial(self.sess.run, result_params, feed_dict=query_params),
        )

    def list_signatures(self):
        signatures = []
        signature_def_map = self.graph_def.signature_def
        for key, signature_def in signature_def_map.items():
            signature = {}
            signature["name"] = key
            signature["inputs"] = {}
            signature["outputs"] = {}
            for key, tensor_info in signature_def.inputs.items():
                signature["inputs"][key] = {
                    "type": tf.as_dtype(tensor_info.dtype).name
                    if tensor_info.dtype
                    else "unknown",
                    "shape": "unkown"
                    if tensor_info.tensor_shape.unknown_rank
                    else [dim.size for dim in tensor_info.tensor_shape.dim],
                }
            for key, tensor_info in signature_def.outputs.items():
                signature["outputs"][key] = {
                    "type": tf.as_dtype(tensor_info.dtype).name
                    if tensor_info.dtype
                    else "unknown",
                    "shape": "unkown"
                    if tensor_info.tensor_shape.unknown_rank
                    else [dim.size for dim in tensor_info.tensor_shape.dim],
                }
            signatures.append(signature)
        return signatures
