import { useEffect } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/dist/ScrollTrigger'

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger)
}

/**
 * Scrollytelling hook: pins the chart wrapper and toggles step opacity.
 * Mirrors the vanilla JS ScrollTrigger scrollytelling pattern.
 *
 * @param {React.RefObject} sectionRef - ref to the .scrolly-section wrapper
 */
export function useScrollReveal(sectionRef) {
  useEffect(() => {
    if (!sectionRef.current) return

    const section = sectionRef.current
    const chartWrapper = section.querySelector('.chart-wrapper')
    const steps = section.querySelectorAll('.step')
    const lastStep = steps[steps.length - 1]

    if (!chartWrapper || steps.length === 0) return

    // Pin the chart wrapper while steps scroll past it
    const pinTrigger = ScrollTrigger.create({
      trigger: chartWrapper,
      endTrigger: lastStep,
      start: 'top top+=80',
      end: () => {
        const height = window.innerHeight
        const chartHeight = chartWrapper.offsetHeight
        return `bottom ${chartHeight + (height - chartHeight) / 2}px`
      },
      pin: true,
      pinSpacing: false,
    })

    // Toggle opacity on each step when it enters/leaves viewport
    const stepTriggers = []
    steps.forEach(step => {
      const st = ScrollTrigger.create({
        trigger: step,
        start: 'top 80%',
        end: 'center top',
        toggleClass: 'active',
      })
      stepTriggers.push(st)
    })

    return () => {
      pinTrigger.kill()
      stepTriggers.forEach(t => t.kill())
    }
  }, [sectionRef])
}
