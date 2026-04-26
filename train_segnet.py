from train_hybrid import build_parser, train_model


def main() -> None:
    parser = build_parser()
    parser.set_defaults(model_type="segnet", output_dir="outputs", pretrained=False)
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
