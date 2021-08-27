from scgenerator import Parameters
import os


def main():
    cwd = os.getcwd()
    try:
        os.chdir("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations")

        pa = Parameters.load("PM1550+PM2000D/PM1550_PM2000D raman_test/initial_config_0.toml")

        print(pa)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
