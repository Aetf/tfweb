import sys
import argparse
from contextlib import contextmanager
from pathlib import Path

import asyncio
import uvloop

import tensorflow as tf

from aiohttp import web
import aiohttp_cors

from model import Model
from batcher import Batcher

from json_handler import JsonHandler
from grpc_handler import GrpcHandler

from grpclib.server import Server


async def on_shutdown(app):
    for task in asyncio.Task.all_tasks():
        task.cancel()


async def init(loop, args):
    if args.tags:
        tags = args.tags.split(',')
    else:
        tags = [Model.default_tag]
    config = tf.ConfigProto()
    config.salus_options.resource_map.persistant["SCHED:PRIORITY"] = 10
    model = Model(args.model, tags, loop, sess_args={
            'target': args.sess_target,
            'config': config
    })
    batcher = Batcher(model, loop, args.batch_size)

    web_app = web.Application(loop=loop, client_max_size=args.request_size)
    web_app.on_shutdown.append(on_shutdown)
    web_app.router.add_get('/stats', batcher.stats_handler)

    json_handler = JsonHandler(model, batcher, args.batch_transpose)

    if args.no_cors:
        web_app.router.add_get('/', batcher.info_handler)
        web_app.router.add_post('/{method}', json_handler.handler)
        web_app.router.add_post('/', json_handler.handler)
    else:
        cors = aiohttp_cors.setup(
                web_app,
                defaults={
                        "*":
                        aiohttp_cors.ResourceOptions(
                                allow_credentials=True,
                                expose_headers="*",
                                allow_headers="*")
                })

        get_resource = cors.add(web_app.router.add_resource('/'))
        cors.add(get_resource.add_route("GET", batcher.info_handler))

        post_resource = cors.add(web_app.router.add_resource('/{method}'))
        cors.add(post_resource.add_route("POST", json_handler.handler))

        post_resource = cors.add(web_app.router.add_resource('/'))
        cors.add(post_resource.add_route("POST", json_handler.handler))

    if args.static_path:
        web_app.router.add_static(
                '/web/', path=args.static_path, name='static')

    grpc_app = Server([GrpcHandler(model, batcher)], loop=loop)

    return web_app, grpc_app


@contextmanager
def pidfile(args):
    if args.listen == '0.0.0.0':
        import socket
        ip = socket.gethostbyname(socket.gethostname())
    else:
        ip = args.listen
    port = args.port
    path = Path(args.piddir).joinpath('{}:{}'.format(ip, port))
    try:
        path.parent.mkdir(exist_ok=True, parents=True)
        # TODO: use real pid
        path.write_text('123')
        yield
    finally:
        path.unlink()


def main(args):
    parser = argparse.ArgumentParser(description='tfweb')
    parser.add_argument(
            '--model',
            type=str,
            default='./examples/basic/model',
            help='path to saved_model directory (can be GCS)')
    parser.add_argument(
            '--tags',
            type=str,
            default=None,
            help='Comma separated SavedModel tags. Defaults to `serve`')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            help='Maximum batch size for batchable methods')
    parser.add_argument(
            '--static_path',
            type=str,
            default=None,
            help='Path to static content, eg. html files served on GET')
    parser.add_argument(
            '--batch_transpose',
            action='store_true',
            help='Provide and return each example in batches separately')
    parser.add_argument(
            '--no_cors',
            action='store_true',
            help='Accept HTTP requests from all domains')
    parser.add_argument(
            '--request_size',
            type=int,
            default=10 * 1024**2,
            help='Max size per request')
    parser.add_argument(
            '--grpc_port',
            type=int,
            default=50051,
            help='Port accepting grpc requests')
    parser.add_argument(
            '--grpc',
            action='store_true',
            help='Enable gRPC server')
    parser.add_argument(
            '--port',
            type=int,
            default=8080,
            help='tfweb model access port')
    parser.add_argument(
            '-l', '--listen',
            type=str,
            default='0.0.0.0',
            help='IP address to listen')
    parser.add_argument(
            '--sess_target',
            type=str,
            default='zrpc://tcp://localhost:5501',
            help='session target for executing inference jobs')
    parser.add_argument(
            '--piddir',
            type=str,
            help='write to a pidfile under the given directory')
    args = parser.parse_args(args)

    def serve():
        uvloop.install()
        loop = asyncio.get_event_loop()

        web_app, grpc_app = loop.run_until_complete(init(loop, args))

        if args.grpc:
                loop.run_until_complete(grpc_app.start(args.listen, args.grpc_port))

        try:
                web.run_app(web_app, host=args.listen, port=args.port)
        except asyncio.CancelledError:
                pass

    if args.piddir is not None:
        with pidfile(args):
            serve()
    else:
        serve()


if __name__ == '__main__':
    main(args=sys.argv[1:])
