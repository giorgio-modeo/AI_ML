from tkinter import filedialog 

nome_file=filedialog.askopenfilename(filetypes=(("File di testo","*.txt"),("Tutti i file ","*.*")))
virus_list = open("p1/virus-list.yml", mode="r",encoding="utf-8")
file= open(nome_file,mode="r",encoding="utf-8")
print(virus_list.read(),file.read())

def e():
  if (e !=re(a)):
    n.removeEventListener(s.trigger, i);
    return
  }
  if (_e(a, e)):
    return
  }
  if (l || Ve(e, a)):
    e.preventDefault()
  }
  if (We(s, e)):
    return
  }
  var t = K(e);
  t.triggerSpec = s;
  if (t.handledFor == null):
    t.handledFor = []
  }
  var r = K(a);
  if (t.handledFor.indexOf(a) < 0):
    t.handledFor.push(a);
    if (s.consume):
      e.stopPropagation()
    }
    if (s.target && e.target):
      if (!d(e.target, s.target)):
        return
      }
    }
    if (s.once):
      if (r.triggeredOnce):
        return
      } else:
        r.triggeredOnce = true
      }
    }
    if (s.changed):
      if (r.lastValue === a.value):
        return
      } else:
        r.lastValue = a.value
      }
    }
    if (r.delayed):
      clearTimeout(r.delayed)
    }
    if (r.throttle):
      return
    }
    if (s.throttle):
      if (!r.throttle):
        o(a, e);
        r.throttle = setTimeout(function():
          r.throttle = null
        }, s.throttle)
      }
    } else if (s.delay):
      r.delayed = setTimeout(function():
        o(a, e)
      }, s.delay)
    } else:
      o(a, e)
    }
  }
}