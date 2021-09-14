import yaml

from music_ai.train import train


def main():
    print("This is going to classify music!")
    # load yaml config
    with open("train_config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    train(cfg)


if __name__ == "__main__":
    main()
