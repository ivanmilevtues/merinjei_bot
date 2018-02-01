function subscribeToComments(btn) {
    jqBtn = $(btn)
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
                window.location.replace("/account/login");
            }
        })
    }
);