document.addEventListener('DOMContentLoaded', function() {
    
    document.querySelector('input').autofocus = true;

    // First Message.
    const first_message = document.createElement('li');
    first_message.innerHTML = "Hi!, I'm Mike I recommend bollywood movies when you say <i>'movies like rockstar'</i>, click 'help' to check out my other features!";

    let loader_element = document.createElement('div')
    loader_element.setAttribute("id", "loader")
    loader_element.style.display = "block"

    // Add class for styling.
    first_message.setAttribute("class", "bot-message");
    
    setTimeout(function () {
        loader_element.style.display = "none"
        // Bot Message bubble.
        document.querySelector('ul').append(first_message);
        return false;
    }, 1000)

    // Help floating window.
    var modal = document.getElementById("help-window");
    var btn = document.getElementById("help-button");
    var span = document.getElementsByClassName("close")[0];

    btn.onclick = function() {
    modal.style.display = "block";
    }

    span.onclick = function() {
    modal.style.display = "none";
    }

    window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
    }

    // Chat Box.
    document.querySelector('form').onsubmit = function() {

        // Checking for blank submission.
        let blank_submit = document.querySelector('input').value

        if (blank_submit == "" || blank_submit == null) {  //Blank submit check.
            alert("Blank Submission was made")
        }
        else if(/^\s*$/.test(blank_submit)) {   // Blank Lines Check.
            alert("Blank Submission was made")
        }
        
        else { 
            // Handle User input.
            let user_input = document.querySelector('input').value;
            console.log(user_input)

            loader_element.style.display = "block"
            document.querySelector("#load").appendChild(loader_element)
            
            // Send POST to server.
            const data = { user_input: user_input };

            fetch('/chat', {
            method: 'POST',
            
            headers: {
                'Content-Type': 'application/json',
            },
            
            body: JSON.stringify(data),
            })
                .then(response => response.json())
                
                .then(data => {
                    console.log('Success:', data);
                    // Bot Response.
                    let bot_response = document.createElement('li')
                    var bot_message =  data["bot_response"]["response"]
                    var intent_detected = data["bot_response"]["intent_detected"]
                    var confidence = data["bot_response"]["confidence"]
                    var unk_percent = data["bot_response"]["unk_percent"]
                    var span_message = "Intent: "+intent_detected+", Confidence: "+confidence+"%, Unk percent: "+unk_percent+"%"

                    var bot_response_extra = "<div class='response-extra'><hr style='width:20%;'>" +span_message+"</div>"

                    bot_response.innerHTML = bot_message.concat(bot_response_extra)
                    bot_response.setAttribute("class", "bot-message")
                    
                    setTimeout(function () {
                        loader_element.style.display = "none"
                        // Bot Message bubble.
                        document.querySelector('ul').append(bot_response);
                        return false;
                    }, 1000)
                })
                .catch((error) => {console.error('Error:', error)});
            

            // User Message bubble.
            const user_response = document.createElement('li');
            user_response.innerHTML = user_input;

            // Add class for styling.
            user_response.setAttribute("class", "user-message");
            document.querySelector('ul').append(user_response);

            // Empty form.
            document.querySelector('input').value = ''

            return false;
        }
    }
})