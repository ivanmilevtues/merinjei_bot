class FileOpenerMixin:
    
    def open_files(self, paths=None):
        main_dir, sub_directories, file_names = paths['main_dir'], paths['sub_directories'], paths['file_names']
        files = []
        for sub_dir in sub_directories:
            for file_name in file_names:
                path = main_dir + "/" + sub_dir + "/" + file_name
                path = self.__generate_file_path(path)
                files.append(open(path))
        return files

    def close_files(self, files):
        for file in files:
            file.close()

    def __generate_file_path(self, path):
        return '/'.join(path.split('//'))