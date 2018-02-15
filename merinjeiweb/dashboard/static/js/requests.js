function handleCommentBtn(btn) {
    jqBtn = $(btn)
    if (jqBtn.html().indexOf("ok") != -1) {
        unsubscribeToComments(jqBtn);
    } else {
        subscribeToComments(jqBtn);
    }
}


function handleMessengerBtn(btn) {
    jqBtn = $(btn)
    if (jqBtn.html().indexOf("ok") != -1) {
        unsubscribeToMessenger(jqBtn);
    } else {
        subscribeToMessenger(jqBtn);
    }
}


function unsubscribeToComments(jqBtn) {
    var pageId = jqBtn.attr('data-id');
    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()

    $.ajax({
        type: "POST",
        url: "/hatespeech/unsubscribe/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
        },
        success: function () {
            jqBtn.empty();
            jqBtn.append('<i class="glyphicon glyphicon-remove"></i>');
        }
    });
}


function unsubscribeToMessenger(jqBtn) {
    var pageId = jqBtn.attr('data-id');
    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()

    $.ajax({
        type: "POST",
        url: "/chatbot/unsubscribe/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
        },
        success: function () {
            jqBtn.empty();
            jqBtn.append('<i class="glyphicon glyphicon-remove"></i>');
        }
    });
}


function subscribeToComments(jqBtn) {
    var pageId = jqBtn.attr('data-id');
    var accessToken = jqBtn.attr('data-token');

    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()
    // Call for full scan and remove hatespeech
    $.ajax({
        type: "POST",
        url: "/hatespeech/scan_for_hatespeech/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
            "access_token": accessToken
        },
    });

    // Subscribe to the webhook for the current page
    $.ajax({
        type: "POST",
        url: "/hatespeech/subscribe/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
            "access_token": accessToken
        },
        success: function() {
            console.log('I am here???');
            jqBtn.empty();
            jqBtn.append('<i class="glyphicon glyphicon-ok"></i>');
        }
    });
}

function subscribeToMessenger(btn) {
    jqBtn = $(btn)
    var pageId = jqBtn.attr('data-id');
    var accessToken = jqBtn.attr('data-token');

    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()
    // Subscribe to messenger notifications for the current page
    $.ajax({
        type: "POST",
        url: "/chatbot/subscribe/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
            "access_token": accessToken
        },
        success: function(result) {
            console.log('I am here?>');
            jqBtn.empty();
            jqBtn.append('<i class="glyphicon glyphicon-ok"></i>');
        }
    });
}


$('#logout a').click(
    function() {
        var CSRF = $('input[name="csrfmiddlewaretoken"]').val()
        $.ajax({
            type: "POST",
            url: "/accounts/logout/",
            data: {
                csrfmiddlewaretoken: CSRF
            },
            success: function(result) {
                window.location.replace("/logged/login");
            }
        })
    }
);
