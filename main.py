from collections import UserDict
from datetime import datetime, date, timedelta
import pickle
from pathlib import Path
from abc import ABC, abstractmethod


class Basic(ABC):
    @abstractmethod
    def is_valid(self, value):
        pass

class Field(Basic):
    
    def __init__(self, value):
        if not self.is_valid(value):
            raise ValueError
        self._value = value


    def is_valid(self, value):
        return True


    @property
    def value(self):
        return self._value
    

    @value.setter
    def value(self, value):
        if not self.is_valid(value):
            raise ValueError('incorrect value')
        self._value = value
    

    def __str__(self):
        return str(self.value)


class Name(Field):
    
    def is_valid(self, value):
        if value.isalpha() :
            return True  
        else:
            return False


class Phone(Field):
    
    def is_valid(self, value):
        if value.isnumeric() and len(value) == 10:
            return True  
        else:
            return False


class Birthday(Field):
    
    def is_valid(self, value):
        if  datetime.strptime(value, r"%d-%m-%Y"):
            return True
        else:  
            return False   
       

class BasicRecord(ABC):
    @abstractmethod
    def add(self, value):
        pass

    @abstractmethod
    def remove(self, value):
        pass
    
    @abstractmethod
    def edit(self, value, new_value):
        pass

    @abstractmethod
    def find(self, value):
        pass


class Record(BasicRecord):
    def __init__(self, name):
        self.name = Name(name)
        self.phones = []

    def add(self, phone):
        phone = Phone(phone)
        self.phones.append(phone) 
        return self.phones    
    
    def remove(self, phone):
        for tel in self.phones:
            if tel.value == phone:
                self.phones.remove(tel)    
    
    def edit(self, phone, new_phone): 
        for tel in self.phones:
            if tel.value == phone:
                new_phone = Phone(new_phone)
                ind = self.phones.index(tel)
                self.phones.insert(ind, new_phone)
                self.phones.remove(tel)
                return
        print('there is not the number in the book ')    
        raise ValueError    
    
    def find(self, phone):
        for tel in self.phones:
            if tel.value == phone:
                return tel
                    
    def days_to_birthday(self, birthday):
        t_date = date.today()
        birth = (datetime.strptime(birthday.date, r"%d-%m-%Y")).date()
        birth = birth.replace(year = t_date.year)
        if t_date <  birth:
            quantity = birth - t_date
        else:
            birth = birth.replace(year = t_date.year+1)
            quantity = birth - t_date
        return quantity.days

    def get_phones(self):
        return self.phones
    
    def __str__(self):
        return f"Contact name: {self.name.value}, phones: {'; '.join(p.value for p in self.phones)}"    


class AddressBook(UserDict):
    
    def list_contacts(self):
        output = []
        for el in self.data.values():
            output.append(str(el))
        return output

    def add_record(self, record):
        contact = record.name
        self.data[contact] = record   
      
    def find(self, contact):
        if contact in self.data:
            return self.data[contact] 
        else:
            return None
        
    def delete(self, name):
        if name in self.data:
            self.data.pop(name)
        else:
            return None
        
    def iterator(self, N):
            list1 = list(self.data.values())
            for i in range(0, len(list1), N):
                yield list1[i:i+N]

    def save_contacts_to_file(self,filename):
        with open(filename, "wb") as fh:
            pickle.dump(self, fh) 
            
    def read_contacts_from_file(filename):
        with open(filename, "rb") as fh:
            unpacked = pickle.load(fh)
        return unpacked
    
    def find_inf(self, inf: str):
        output = []
        if inf.isalpha():
            for key, el in self.data.items():
                x = key.value
                if x.find(inf) != -1:
                    output.append(str(el))
        elif inf.isdigit():
            for key, el in self.data.items():
                for tel in el.phones:
                    x = tel.value
                    if x.find(inf) != -1:
                        output.append(str(el))
                        break   
        else:
            raise Exception("Sorry, incorrect input")                  
        if len(output) == 0:
            return 'There are no contacts with such input'
        return output


def input_error(func):
    def inner_Function(contact_book, data):
        if data == '':
            func_arg = ''
        else:
            func_arg = data.strip().split(' ')
    
        try:
            res = func(contact_book, *func_arg) 
        except:
            print("incorrect arguments")
            res = None
        return res    
    return inner_Function


@input_error
def add(contact_book, name, phone):
    for el in contact_book.data.values():
        if name == el.name.value:
            el.add(phone)
            return 'added'
    new_contact = Record(name)
    contact_book.add_record(new_contact)
    contact_book[new_contact.name].add(phone)
    print(str(new_contact))
    return 'created'


@input_error
def phone(contact_book, phone):
    if phone.isdigit():
        return contact_book.find_inf(phone)
    else:
        return None

@input_error
def change(contact_book, phone, new_phone):
    for phone_contact_book in contact_book.data.values():
        for tel in phone_contact_book.phones:
            if phone == tel.value:
                phone_contact_book.edit(phone, new_phone)
                return 'changed'
    return 'there is not such number in the contact_book '       


@input_error
def show(contact_book):
    return contact_book.list_contacts()


@input_error
def find_list(contact_book, inf):
    return contact_book.find_inf(inf)


@input_error
def close(contact_book):
    contact_book.save_contacts_to_file('storage.bin')
    return 'close'


@input_error
def hello(contact_book):
    return 'Hello! How can I help you?'


action = {'hello': hello, 'add': add, 'change': change, 'phone': phone, 'show all': show,
             'good bye': close, 'exit': close, 'close': close, 'find': find_list}


def choice(data: str):
    i = False
    for command in action:
        if data.startswith(command):
            line = data.removeprefix(command)
            i = True
            return action[command], line
        else:
            continue
    if not i:
        print('Sorry, there is not such command')
    return 'Give a command, please'  , ''  


def start():
    if Path('storage.bin').is_file():
        return AddressBook.read_contacts_from_file('storage.bin')
    else:
        return AddressBook()


def do_work():
    start()
    contact_book = start()
    while True:
        data = input('Enter command  ')
        command, line = choice(data)
        if isinstance(command, str):
            print(command)
            continue
        result = command(contact_book, line)
        print(result)
        if result == 'close':
            break


if __name__ == '__main__':
    do_work()