import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
 

def send_email():
    sender_email_address = 'parking.recognition@gmail.com'
    sender_email_password = 'parking.recognition12!@'
    receiver_email_address = 'parking.recognition.client@gmail.com'
    
    email_subject_line = 'INCORRECT PARKING'
    
    msg = MIMEMultipart()
    msg['From'] = sender_email_address
    msg['To'] = receiver_email_address
    msg['Subject'] = email_subject_line
    
    email_body = 'This is a notification that the car in spot 1 has been incorrectly parked.\n'
    msg.attach(MIMEText(email_body, 'plain'))
    
    email_content = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(sender_email_address, sender_email_password)
    
    server.sendmail(sender_email_address, receiver_email_address, email_content)
    server.quit()