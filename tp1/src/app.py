import argparse as arg
from .model import driver


class Args:
    def __init__(self, hidden: int, rate: int, batch: int) -> None:
        self.hidden = hidden
        self.rate = rate
        self.batch = batch


def arg_parser():
    parser: arg.ArgumentParser = arg.ArgumentParser(
        description="Monta uma rede neural para resolver o dataset MNIST."
    )

    parser.add_argument(
        "--hidden",
        dest="hidden",
        required=True,
        type=int,
        help="Número de unidades na camada oculta.",
    )

    parser.add_argument(
        "--rate",
        dest="rate",
        required=True,
        type=float,
        help="Taxa de aprendizado.",
    )

    parser.add_argument(
        "--batch",
        dest="batch",
        required=True,
        type=int,
        help="Tamanho do Batch. Use para controlar qual algoritmo será usado.",
    )
    args = parser.parse_args()

    return args.hidden, args.rate, args.batch


def run():
    args = Args(*arg_parser())
    driver(args.hidden, args.rate, args.batch)
