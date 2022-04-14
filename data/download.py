from lib.utils.config import Config
import os
import argparse
import glob
import gdown


def download_and_unzip(source_url, zip_path, local_path, use_tar=False, http=False):
    if not os.path.exists(os.path.dirname(zip_path)):
        os.makedirs(os.path.dirname(zip_path))
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    print("Download from {} to {}".format(source_url, zip_path))
    command = "wget -O {} '{}'".format(zip_path, source_url)
    if http:
        command += " --no-check-certificate"
    os.system(command)
    print("Download done!")

    print("Unzip from {} to {}".format(zip_path, local_path))
    if use_tar:
        unzip_cmd = "tar xf {} -C {}".format(zip_path, local_path)
    else:
        unzip_cmd = "unzip {} -d {}".format(zip_path, local_path)
    os.system(unzip_cmd)
    print("Unzip done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Template Matching download scripts')
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    config = Config(config_file=args.config).get_config()

    if args.dataset in ["linemod", "all"]:
        # download LINEMOD dataset and unzip
        download_and_unzip(config.LINEMOD.source_url, os.path.join(config.root_path, "zip/linemod.zip"),
                           os.path.join(config.root_path, config.LINEMOD.local_path))

        # download OcclusionLINEMOD dataset and unzip
        download_and_unzip(config.occlusionLINEMOD.source_url,
                           os.path.join(config.root_path, "zip/occlusionLinemod.zip"),
                           os.path.join(config.root_path, config.occlusionLINEMOD.local_path))

        refactor_command = "mv {} {}".format(os.path.join(config.root_path, config.occlusionLINEMOD.local_path,
                                                          "OcclusionChallengeICCV2015/*"),
                                             os.path.join(config.root_path, config.occlusionLINEMOD.local_path))
        os.system(refactor_command)

        remove_command = os.path.join(config.root_path, config.occlusionLINEMOD.local_path,
                                      "OcclusionChallengeICCV2015/*")
        os.system(remove_command)

        # download CAD models of LINEMOD
        download_and_unzip(config.LINEMOD.cad_url, os.path.join(config.root_path, "zip/models.zip"),
                           os.path.join(config.root_path, config.LINEMOD.cad_path), http=True)

    if args.dataset in ["tless", "all"]:
        # download training scenes of T-LESS dataset from BOP challenge
        url = config.TLESS.source_url + "/tless_train_primesense.zip"
        local = os.path.join(config.root_path, config.TLESS.local_path)
        zip_name = os.path.join(config.root_path, "zip/tless_train_primesense.zip")
        download_and_unzip(url, zip_name, local, http=True)
        # refactor "tless/train_primesense" to "tless/train"
        os.system("mv {} {}".format(os.path.join(local, "train_primesense"), os.path.join(local, "train")))

        # download 20 test scenes of T-LESS dataset from BOP challenge
        url = config.TLESS.source_url + "/tless_test_primesense_all.zip"
        local = os.path.join(config.root_path, config.TLESS.local_path)
        zip_name = os.path.join(config.root_path, "zip/tless_test_primesense_all.zip")
        download_and_unzip(url, zip_name, local, http=True)
        # refactor "tless/test_primesense" to "tless/test"
        os.system("mv {} {}".format(os.path.join(local, "test_primesense"), os.path.join(local, "test")))

        # download camera intrinsic for test set and list of all test images
        url = config.TLESS.source_url + "/tless_base.zip"
        tmp_path = os.path.join(config.root_path, "zip")
        local = os.path.join(config.root_path, config.TLESS.local_path)
        zip_name = os.path.join(config.root_path, "zip/tless_base.zip")
        download_and_unzip(url, zip_name, tmp_path, http=True)
        os.system("mv {} {}".format(os.path.join(tmp_path, "tless/*"), os.path.join(local)))

        # download CAD models of TLess
        download_and_unzip(config.TLESS.cad_url + "/t-less_v2_models_cad.zip",
                           os.path.join(config.root_path, "zip/models_cad.zip"),
                           os.path.join(config.root_path, config.TLESS.cad_path))

        download_and_unzip(config.TLESS.cad_url + "/t-less_v2_models_eval.zip",
                           os.path.join(config.root_path, "zip/models_eval.zip"),
                           os.path.join(config.root_path, config.TLESS.cad_path))

    if args.dataset in ["preprocessed_in_gdrive"]:
        # download datasets from gdrive
        for name in config.gdrive.keys():
            if not os.path.exists(os.path.join(config.root_path, "zip")):
                os.makedirs(os.path.join(config.root_path, "zip"))
            zip_path = os.path.join(config.root_path, "zip", "{}.zip".format(name))
            gdown.download(config.gdrive[name], zip_path, quiet=False, fuzzy=True)
            if name in ["linemod", "templates", "TLESS"]:
                unzip_command = "unzip {} -d {}".format(zip_path, config.root_path)
                os.system(unzip_command)
            else:
                unzip_command = "unzip {} -d {}".format(zip_path, config.root_path)
                os.system(unzip_command)
                dataframe_paths = glob.glob(config.root_path + "/dataframes/*.json")
                for dataframe_path in dataframe_paths:
                    new_dataframe_path = dataframe_path.replace("/dataframes/", "/")
                    refactor_command = "mv {} {}".format(dataframe_path, new_dataframe_path)
                    os.system(refactor_command)
                os.system("rm -r {}".format(config.root_path + "/dataframes"))

    if args.dataset in ["SUN397", "preprocessed_in_gdrive", "all"]:
        # download LINEMOD dataset and unzip
        download_and_unzip(config.SUN397.source_url, os.path.join(config.root_path, "zip/SUN397.zip"),
                           os.path.join(config.root_path, config.SUN397.local_path), use_tar=True)
