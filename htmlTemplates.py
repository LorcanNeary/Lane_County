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
        <img src="https://salanecounty.blob.core.windows.net/sc-bot-images/lane_county_robot.webp?sp=r&st=2025-02-18T21:55:20Z&se=2026-02-19T05:55:20Z&spr=https&sv=2022-11-02&sr=b&sig=ZhGUKRQl2oruXgJvAF0K6YyTsZmafYSY0q4oc4F5qHo%3D" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://salanecounty.blob.core.windows.net/sc-bot-images/lorcan_image.png?sp=r&st=2025-02-18T21:57:23Z&se=2026-02-19T05:57:23Z&spr=https&sv=2022-11-02&sr=b&sig=C3KwHQLWMuRq3NL64Gbdk2tSodnohsnJPmU2l3NJkF0%3D">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
