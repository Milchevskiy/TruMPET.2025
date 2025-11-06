import os

class Scheduler:
    def __init__(self, parm_source_file_name: str):
        self.sheduler_option = {}
        self.is_initialized_yet = False
        self.init(parm_source_file_name)
    
    def init(self, parm_source_file_name: str):
        """Инициализация из файла параметров."""
        if not os.path.exists(parm_source_file_name):
            print(f"Scheduler: file {parm_source_file_name} not found")
            exit(1)
        
        with open(parm_source_file_name, 'r') as parm_stream:
            self.taskset_get_option(parm_stream)
    
    def taskset_get_option(self, parm_stream):
        """Считывает параметры из файла."""
        for line in parm_stream:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                self.sheduler_option[key] = value
    
    def display_sheduler_option_setted(self, path_to_option_show_file_name: str):
        """Выводит сохранённые опции в файл."""
        try:
            with open(path_to_option_show_file_name, 'w') as show_stream:
                for key, value in self.sheduler_option.items():
                    show_stream.write(f"{key:<25} {value:<35}\n")
        except IOError:
            print("Can't create option show file")
            exit(1)
    
    def option_meaning(self, key: str) -> str:
        """Возвращает значение параметра по ключу."""
        return self.sheduler_option.get(key, "UNKNOWN")
    
    def add_key_meaning_record(self, key: str, meaning: str):
        """Добавляет новую запись в параметры."""
        self.sheduler_option[key] = meaning

if __name__ == "__main__":
    scheduler = Scheduler('PB_W7_tail_GP/sheduler')
    scheduler.display_sheduler_option_setted("output.txt")
    
    value=scheduler.option_meaning('CLUSTER_SET_NAME')
    print(f'option_meaning for CLUSTER_SET_NAME:{value}')

    value=scheduler.option_meaning('ASDF')
    print(f'option_meaning for ASDF:{value}')



