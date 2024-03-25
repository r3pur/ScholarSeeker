document.addEventListener("DOMContentLoaded", function () {
    var inputField = document.getElementById("user-input");
    inputField.addEventListener("keypress", function (event) {
        if (event.keyCode === 13) { // Enter key pressed
            event.preventDefault(); // Prevent default form submission
            sendMessage(); // Call sendMessage function
        }
    });
});

document.addEventListener("DOMContentLoaded", function () {
    // Create the reset button element
    var resetButton = document.createElement("button");
    resetButton.id = "reset"; // Set the button's ID for CSS styling and event handling
    resetButton.textContent = "Reset Chat"; // Set the button text

    // Append the reset button to the body or another container element
    document.body.appendChild(resetButton); // This could be another container element as needed

    // Add click event listener for the reset button
    resetButton.addEventListener('click', function () {
        fetch('/reset', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message); // Log the reset confirmation
            var chatBox = document.getElementById('chat-box');
            chatBox.innerHTML = ''; // Clear the chat box
        })
        .catch(error => console.error('Error:', error));
    });
});


document.addEventListener("DOMContentLoaded", function () {
    document.getElementById('uploadButton').addEventListener('click', function () {
        var modal = document.getElementById("uploadButton");
        modal.style.display = "none";
        document.getElementById('fileInput').click();
        var fileInput = document.getElementById("fileInput");
        var file = fileInput.files[0];
        if (file) {
            var formData = new FormData();
            formData.append("file", file);

            fetch("/upload", {
                method: "POST",
                body: formData,
            })
                .then((response) => response.json())
                .then((data) => {
                    console.log(data);
                })
                .catch((error) => console.error("Error:", error));
        }
    });
});

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById('reset').addEventListener('click', function () {
        var modal = document.getElementById("uploadButton");
        var chatBox = document.getElementById('chat-box');
        modal.style.display = "block";
        // Clear the chat box before adding new messages
        chatBox.innerHTML = '';

        // // Make an AJAX request to delete uploaded files
        // fetch("/delete_files", {
        //     method: "POST",
        // })
        //     .then((response) => response.json())
        //     .then((data) => {
        //         console.log(data);
        //     })
        //     .catch((error) => console.error("Error:", error));

    });
});


function sendMessage() {
    // Delay execution of sendMessage function for 100 milliseconds
    setTimeout(function() {
        var uploadButton = document.getElementById('uploadButton');
        var modalDisplayStyle = window.getComputedStyle(uploadButton).getPropertyValue('display');

        // // Check if the upload button is currently displayed
        // if (modalDisplayStyle === 'block') {
        //     // If the upload button is displayed, show an alert message
        //     alert("You first need to upload a file before you can write queries to ScholarSeeker.");
        //     return; // Stop further execution of the function
        // }

        // If the upload button is not displayed, proceed with sending the message
        var userMessage = document.getElementById('user-input').value.trim();
        if (userMessage !== '') {
            displayMessage('You', userMessage);
            document.getElementById('user-input').value = '';
            // Send the user's message to the server for processing
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage })
            })
                .then(response => response.json())
                .then(data => {
                    displayMessage('ScholarSeeker', data.message);
                })
                .catch(error => console.error('Error:', error));
        }
    }, 100); // Delay set to 100 milliseconds
}


function displayMessage(sender, message) {
    var chatBox = document.getElementById('chat-box');

    // Create a new message element
    var newMessage = document.createElement('div');
    newMessage.classList.add(sender);

    // Create a paragraph element for the sender's name
    var senderName = document.createElement('p');
    senderName.textContent = sender + ":";
    senderName.classList.add('who');

    // Create a paragraph element for the message content
    var messageContent = document.createElement('p');
    messageContent.textContent = message;
    messageContent.classList.add('sender');

    // Append the sender's name and message content to the new message element
    newMessage.appendChild(senderName);
    newMessage.appendChild(messageContent);

    // Append the new message to the chat box
    chatBox.appendChild(newMessage);

    // Check if the sender is the bot and if the previous message was from the user
    if (sender === 'ScholarSeeker' && chatBox.lastChild.previousSibling.classList.contains('You')) {
        // Create a new element for displaying the current time
        var timeElement = document.createElement('div');
        timeElement.classList.add('time'); // Add a class for styling (optional)

        // Get the current time
        var currentTime = new Date();
        var hours = currentTime.getHours();
        var minutes = currentTime.getMinutes();
        var ampm = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        hours = hours ? hours : 12; // Handle midnight (0 hours)
        minutes = minutes < 10 ? '0' + minutes : minutes; // Add leading zero for single-digit minutes

        // Set the time text
        var timeText = hours + ':' + minutes + ' ' + ampm;

        // Create a text node with the time text
        var timeTextNode = document.createTextNode(timeText);

        // Append the time text node to the time element
        timeElement.appendChild(timeTextNode);

        // Append the time element to the chat box
        chatBox.appendChild(timeElement);
        // Create and append a delimiter element
        //var delimiter = document.createElement('hr');
        //chatBox.appendChild(delimiter);

        // Create the download output button
        var downloadButton = document.createElement('button');
        downloadButton.textContent = 'Download Tabular Output';
        downloadButton.classList.add('button');
        chatBox.appendChild(downloadButton);

        // Create the view output button
        var viewButton = document.createElement('button');
        viewButton.textContent = 'View Tabular Output';
        viewButton.classList.add('button');
        chatBox.appendChild(viewButton);

    }

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
}
