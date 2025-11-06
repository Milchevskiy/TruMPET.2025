class CurlyCalculusSystem:
    def __init__(self, base):
        if not base:
            raise ValueError("Base list cannot be empty!")
        if any(b <= 0 for b in base):
            raise ValueError("All elements in base must be > 0!")
        
        self.base = base
        self.number_of_elements = 1
        for b in base:
            self.number_of_elements *= b

    def get_cursor_by_array(self, array):
        if len(array) != len(self.base):
            raise ValueError("Array size does not match base size!")
        if any(a < 0 or a >= b for a, b in zip(array, self.base)):
            raise ValueError("Array values must be within valid range.")
        
        temp = self.number_of_elements
        cursor = 0
        
        for a, b in zip(array, self.base):
            temp //= b
            cursor += temp * a
        
        return cursor
    
    def get_array_by_cursor(self, cursor):
        if cursor < 0 or cursor >= self.number_of_elements:
            raise ValueError("Cursor out of valid range!")
        
        array = []
        divisor = self.number_of_elements
        
        for b in self.base:
            divisor //= b
            array.append(cursor // divisor)
            cursor %= divisor
        
        return array

    def get_number_of_elements(self):
       return self.number_of_elements


# Пример использования
if __name__ == "__main__":
    base = [4, 4, 4]  # Пример базового множества
    system = CurlyCalculusSystem(base)
    
    array = [0, 3, 0]
    cursor = system.get_cursor_by_array(array)
    print(f"Cursor for {array}: {cursor}")
    
#    recovered_array = system.get_array_by_cursor(cursor)
#    print(f"Array for cursor {cursor}: {recovered_array}")
