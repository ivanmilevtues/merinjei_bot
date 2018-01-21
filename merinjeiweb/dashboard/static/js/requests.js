function subscribeToComments(btn) {
    var pageUrl = $('#url-input').val();
    jqBtn = $(btn)
    var pageId = jqBtn.attr('data-id');
    var accessToken = jqBtn.attr('data-token');

    var CSRF = $('input[name="csrfmiddlewaretoken"]').val()
 
    $.ajax({
        type: "POST",
        url: "/hatespeech/subscribe/",
        data: {
            csrfmiddlewaretoken: CSRF,
            "page_id": pageId,
            "access_token": accessToken
        },
        success: function(data) {
            $('#comments-container').append(data);
        }
    });
}