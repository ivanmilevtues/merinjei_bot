from django.http import HttpResponseRedirect


def redirect_to_login(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/logged/profile')
    return HttpResponseRedirect('/logged/login')
