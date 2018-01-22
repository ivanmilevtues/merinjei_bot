function subscribeToComments(btn) {
    jqBtn = $(btn)
    var pageId = jqBtn.attr('data-id');
    var accessToken = jqBtn.attr('data-token');

    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()
    $.ajax({
        type: "POST",
        url: "/hatespeech/scan_for_hatespeech/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
            "access_token": accessToken
        },
    });

    $.ajax({
        type: "POST",
        url: "/hatespeech/subscribe/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
            "access_token": accessToken
        },
    });
}

function subscribeToMessenger(btn) {
    jqBtn = $(btn)
    var pageId = jqBtn.attr('data-id');
    var accessToken = jqBtn.attr('data-token');

    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()

    $.ajax({
        type: "POST",
        url: "/chatbot/subscribe/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
            "access_token": accessToken
        },
    });
}