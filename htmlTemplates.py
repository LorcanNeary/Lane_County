css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://salanecounty.blob.core.windows.net/sc-bot-images/lane_county_robot.webp?sp=r&st=2025-02-18T20:32:19Z&se=2025-02-19T04:32:19Z&spr=https&sv=2022-11-02&sr=b&sig=t9Gt1TQJF1u9r4NJp2VvdRdE0zGTJs0%2BVW2Dv9Vs360%3D" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://salanecounty.blob.core.windows.net/sc-bot-images/lorcan_image.png?sp=r&st=2025-02-18T20:34:59Z&se=2025-02-19T04:34:59Z&spr=https&sv=2022-11-02&sr=b&sig=HCcNY6MG93CaHm7l%2Br%2BBu%2BpDmZmiCeUo38hd%2Fj%2F%2Faqg%3D">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
