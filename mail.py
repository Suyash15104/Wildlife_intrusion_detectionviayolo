import smtplib
from email.mime.text import MIMEText

def send_email_alert():
    sender ="suyash.bagale@gmail.com"
    password ="rxkljclybigcrwnu"
    receiver = "suyash@anatechconsultancy.com"

    msg = MIMEText("🚨 Intrusion detected: Human and wildlife detected together.")
    msg["Subject"] = "Intrusion Alert"
    msg["From"] = sender
    msg["To"] = receiver

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()