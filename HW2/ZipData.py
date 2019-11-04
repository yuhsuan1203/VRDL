# +
import os
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

if __name__ == '__main__':
    zipf = zipfile.ZipFile('result_image.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('final_result-1104_151744/', zipf)
    zipf.close()
