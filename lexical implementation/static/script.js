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
    var resetButton = document.getElementById("reset");
    // resetButton.id = "reset";
    // resetButton.textContent = "Reset Chat";
    // document.body.appendChild(resetButton);

    // Add click event listener for the reset button
    resetButton.addEventListener('click', resetChat);
});

function resetChat() {
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
}



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



function sendMessage() {
    // Delay execution of sendMessage function for 100 milliseconds
    setTimeout(function() {
        var uploadButton = document.getElementById('uploadButton');
        var modalDisplayStyle = window.getComputedStyle(uploadButton).getPropertyValue('display');

        var userMessage = document.getElementById('user-input').value.trim();
        if (userMessage !== '') {
            // Attempt to send the user's message to the server for processing
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: userMessage })
            })
            .then(response => {
                // First, check if the response is OK.
                if (response.ok) { // Check if the status code is 200
                    return response.json();
                } else {
                    return response.json().then(data => {
                        throw new Error(data.message || 'Must sign in first'); // Use server-provided message or default
                    });
                }
            })
            .then(data => {
                // After ensuring the response was OK and parsing the JSON, display the user's message and then the server's response.
                document.getElementById('centerImage').style.display = 'none';
                displayMessage('You', userMessage); // Now displaying the user's message only after a successful server response
                document.getElementById('user-input').value = ''; // Clear the input field after successful operation
                displayMessage('ScholarSeeker', data.message);
            })
            .catch(error => {
                // Handle any errors that occur during the fetch or processing.
                alert(error.message); // Use the error message from the rejected promise
                console.error('Error:', error);
            });
        }
    }, 100); // Delay set to 100 milliseconds
    fetchArrayAndDisplay();
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

        // Assuming chatBox is already defined as it's used to append the viewButton
        viewButton.addEventListener('click', function() {
            fetch('/get-table')
                .then(response => response.json())
                .then(data => {
                    // Assuming you want to insert the table directly after the button
                    var tableDiv = document.createElement('div');
                    tableDiv.innerHTML = data.html;
                    tableDiv.classList.add('custom-table-style');
                    chatBox.insertBefore(tableDiv, viewButton.nextSibling);
                })
                .catch(error => console.error('Error:', error));
        });


    }

    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
}

document.addEventListener("DOMContentLoaded", function () {
    // Login form submission
    document.getElementById("loginForm").addEventListener("submit", function(event) {
        event.preventDefault(); // Prevent the default form submission

        // Gather form data
        var formData = new FormData(this); // 'this' refers to the form element
        var object = {};
        formData.forEach((value, key) => {object[key] = value;});
        var jsonData = JSON.stringify(object);

        fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: jsonData
        })
        .then(response => {
            if(response.ok) { // Check if the status code is 200
                return response.json();
            } else {
                throw new Error('Login failed!'); // If not 200, reject the promise
            }
        })
        .then(data => {
            alert('Login successful!');
            // Optionally close the modal here if you're programmatically handling it
            document.getElementById('loginModal').style.display = 'none';
            fetchArrayAndDisplay();
        })
        .catch(error => {
            alert(error.message); // Use the error message from the rejected promise
            console.error('Error:', error);
        });
    });

function fetchArrayAndDisplay() {
    fetch('/get-array')
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Failed to load array data.');
            }
        })
        .then(data => {
            const container = document.createElement('div');
            container.style.position = 'absolute';
            container.style.left = '0';
            container.style.top = '0';
            container.id = 'arrayContainer';
            document.body.appendChild(container);
            for (let i = data.length - 1; i >= 0; i--) {
                const div = document.createElement('div');
                div.textContent = data[i];
                div.className = 'array-item';
                container.appendChild(div);
            }
            // data.forEach(item => {

            //     const div = document.createElement('div');
            //     div.textContent = item;
            //     div.className = 'array-item';
            //     container.appendChild(div);
            // });
        })
        .catch(error => console.error('Error loading the array:', error));
}





    // Registration form submission
    document.getElementById("registerForm").addEventListener("submit", function(event) {
        event.preventDefault();

        var formData = new FormData(this);
        var object = {};
        formData.forEach((value, key) => {object[key] = value;});
        var jsonData = JSON.stringify(object);

        // AJAX request to the registration endpoint
        fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: jsonData
        })
        .then(response => {
            if (response.ok) { // Check if the status code is 200
                return response.json();
            } else {
                return response.json().then(data => {
                    throw new Error(data.message || 'Registration failed!'); // Use server-provided message or default
                });
            }
        })
        .then(data => {
            alert('Registration successful!');
            // Optionally close the modal here
            document.getElementById('registerModal').style.display = 'none';
        })
        .catch(error => {
            alert(error.message); // Use the error message from the rejected promise
            console.error('Error:', error);
        });
    });

    document.getElementById('openLoginModal').addEventListener('click', function() {
        document.getElementById('loginModal').style.display = 'block';
    });
    document.getElementById('closeLoginModal').addEventListener('click', function() {
        document.getElementById('loginModal').style.display = 'none';
    });
    
    document.getElementById('openRegisterModal').addEventListener('click', function() {
        document.getElementById('registerModal').style.display = 'block';
    });
    document.getElementById('closeRegisterModal').addEventListener('click', function() {
        document.getElementById('registerModal').style.display = 'none';
    });
});

