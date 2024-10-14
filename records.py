import sqlite3
import os
import base64
import datetime
import json
import cv2

class database_model:
    def __init__(self):
        self.con = sqlite3.connect('Face_Recoginition.db')
        self.cur = self.con.cursor() 

    def dictfetchall(self,cursor):
        columns =[col[0] for col in cursor.description]
        return [dict(zip(columns,row))
                   for row in cursor.fetchall()]
    
    def create_table(self):
        # self.cur.execute("create table user(id, name, password, secret_key)")
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS Records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                main_image BLOB NOT NULL,
                person_image BLOB NOT NULL,
                detected_human_id VARCHAR(255) NOT NULL,
                position BLOB NOT NULL,
                confidence REAL NOT NUll,
                Datetime datetime
            )
            ''')

        self.con.commit()

    def alter_table(self):
        self.cur.execute("alter table user add secret_key")
        self.con.commit()
    
    def drop_table(self):
        self.cur.execute("drop table user_db")
        self.con.commit()
    
    def delete_table(self):
        self.cur.execute("delete from user")
        self.con.commit()
    
    def insert_data(self, main_image, detected_human_id, position, confidence): 
        x,y,w,h = position
        person_image = main_image[y:y + h, x:x + w]
        position_str = json.dumps(position)
        # Convert the images (main and person) to byte strings
        _, main_image_encoded = cv2.imencode('.jpg', main_image)
        _, person_image_encoded = cv2.imencode('.jpg', person_image)
        main_image_bytes = main_image_encoded.tobytes()
        person_image_bytes = person_image_encoded.tobytes()
        Datetime = datetime.datetime.now()
        self.cur.execute("Insert into Records (main_image, person_image, detected_human_id, position, confidence, Datetime) values(?,?,?,?,?,?)",(main_image_bytes,person_image_bytes,detected_human_id, position_str, confidence, Datetime))
        self.con.commit()
    
    def get_data(self):
        self.cur.execute("select * from user")
        data = self.dictfetchall(self.cur)
        if len(data) !=0:
            return data
        return None
        
    
    
# obj= database_model()
# obj.create_table()
# obj.insert_data()
# obj.drop_table()
# obj.get_data()
# obj.delete_table()
# obj.insert_data()
# obj.update_password('kos@123')

