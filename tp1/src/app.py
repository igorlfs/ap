import argparse as arg
from .model import driver


def arg_parser():
    parser: arg.ArgumentParser = arg.ArgumentParser(
        description="Gera um modelo do MNIST"
    )

    parser.add_argument(
        "--hidden-layer",
        dest="hidden",
        required=True,
        type=int,
        help="Número de unidades na camada oculta",
    )

    parser.add_argument(
        "--rate",
        dest="rate",
        required=True,
        type=float,
        help="Taxa de aprendizado",
    )

    parser.add_argument(
        "--grad",
        dest="grad",
        required=True,
        type=str,
        help="Algoritmo de cálculo de gradiente",
    )

    parser.add_argument(
        "--batch",
        dest="batch",
        required=False,
        type=int,
        help="Tamanho do mini-batch",
    )
    args = parser.parse_args()

    if (args.grad == "mini") and (args.batch is None):
        parser.error("--batch é obrigatório para o algoritmo de mini-batch")

    return args


def run() -> None:
    args = arg_parser()
    driver(args.hidden, args.rate, args.grad, args.batch)
